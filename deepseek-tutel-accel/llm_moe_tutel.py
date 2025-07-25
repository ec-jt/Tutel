#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, sys, math
import torch
import pathlib, json, time
import argparse
import numpy as np
from torch.utils.cpp_extension import IS_HIP_EXTENSION

os.environ['TUTEL_GLOBAL_TIMEOUT_SEC'] = str(2147483647)
os.environ['D3D12_ENABLE_FP16'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--try_path', action='append', default=[])
parser.add_argument('--max_seq_len', type=int, default=1024 * 8)
parser.add_argument('--buffer_size', type=int, default=32)
parser.add_argument('--listen_port', type=int, default=8000)
parser.add_argument('--serve', default=False, action='store_true')
parser.add_argument('--prompt', type=str, default='')
parser.add_argument('--disable_thinking', default=False, action='store_true')
parser.add_argument('--disable_fp4', default=False, action='store_true')
args = parser.parse_args()

try:
    from safetensors.torch import safe_open, save_file
    from transformers import AutoTokenizer
except:
    raise Exception(f'Failed to import huggingface, please install the client with:\n\n  >> {sys.executable} -m pip install "transformers" "safetensors"')
    exit(0)

torch.backends.cuda.matmul.allow_tf32 = True

try:
  from tutel import system, net

  if 'OP_LOADER' not in os.environ:
    if not IS_HIP_EXTENSION:
      if torch.cuda.get_device_capability()[0] == 6:
        suffix = 'p100'
      elif torch.cuda.get_device_capability()[0] == 7:
        suffix = 'v100'
      elif torch.cuda.get_device_capability()[0] == 8:
        suffix = 'a100'
      elif torch.cuda.get_device_capability()[0] == 9:
        suffix = 'h100'
      elif torch.cuda.get_device_capability()[0] == 10:
        suffix = 'b200'
      else:
        raise Exception('Device compat not recognized: %s' % torch.cuda.get_device_capability())
    else:
      if ('MI50' in torch.cuda.get_device_name()):
        suffix = 'mi50'
      else:
        suffix = 'mi300x'
    os.environ['OP_LOADER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ops.{suffix}")

  from tutel import ops
  op_loader_path = os.environ['OP_LOADER']

  if not os.path.exists(op_loader_path):
      raise Exception(f'Antares kernels are not properly configured for OP_LOADER at: {op_loader_path}')
  parallel_env = system.init_data_model_parallel(backend='gloo')
  world_rank, world_size, master_print = parallel_env.global_rank, parallel_env.global_size, parallel_env.dist_print
  device = torch.device('cuda', parallel_env.local_rank)
  torch.cuda.set_device(device)
  peer_barrier = lambda: net.simple_all_reduce(torch.ones([1], dtype=torch.int32))

  if os.environ.get('LD_PRELOAD', '').strip():
    from tutel.impls.communicate import init_extern_nccl
    init_extern_nccl()

  def fetch_url(path, url):
    peer_barrier()
    if world_rank == 0:
      system.from_url(url, path=path)
    peer_barrier()
    return torch.from_numpy(np.load(path))
except:
  assert int(os.environ.get('WORLD_SIZE', 1)) == 1, "Failed to initialize distributed session"
  import autort
  world_rank, world_size, master_print = 0, 1, print
  device = autort.device()
  peer_barrier = lambda: None

  def fetch_url(path, url):
    return autort.from_npy(autort.download(path, url))

model_id = None
for path in args.try_path:
  if os.path.exists(path):
    model_id = path
    break

if model_id is None:
    model_id = 'nvidia/DeepSeek-R1-FP4' if len(args.try_path) == 0 else args.try_path[0]
    raise Exception(f"DeepSeek R1/V3/R1-FP4 model data is not found in {model_id}, please download it first:\n\n  >> huggingface-cli download {model_id} --local-dir '{model_id}'\n")
else:
    master_print(f"[INFO] Discover the model from local path: {model_id}, chosen as the default model.")

tokenizer = AutoTokenizer.from_pretrained(f'{model_id}', trust_remote_code=True)
config = json.loads(pathlib.Path(f'{model_id}/config.json').read_text())

model_type = config['model_type']
eos_token_id = tokenizer.eos_token_id if model_type != 'kimi_k2' else 163586
data_type = torch.bfloat16

if model_type in ('deepseek_v3', 'kimi_k2',):
  n_layers = int(os.environ.get('LAYER', config['num_hidden_layers']))
  n_experts = int(os.environ.get('NE', config["n_routed_experts"]))
  n_heads = config['num_attention_heads']
  log_rope_theta = math.log(config['rope_theta'])
  qk_nope_head_dim = config["qk_nope_head_dim"]
  qk_rope_head_dim = config["qk_rope_head_dim"]
  q_lora_rank = config["q_lora_rank"]
  kv_lora_rank = config["kv_lora_rank"]
  v_head_dim = config["v_head_dim"]
  n_top_k = config['num_experts_per_tok']
elif model_type in ('qwen3_moe', 'qwen3',):
  n_layers = int(os.environ.get('LAYER', config['num_hidden_layers']))
  n_experts = int(os.environ.get('NE', config.get("num_experts", 0)))
  n_heads = config['num_attention_heads']
  log_rope_theta = math.log(config['rope_theta'])
  n_top_k = config.get('num_experts_per_tok', 0)
  head_dim = config['head_dim']
  num_key_value_heads = config['num_key_value_heads']
else:
  raise Exception(f'Unrecognized model type: {model_type}')


def load_to(filename, params):
  with safe_open(f'{filename}', framework='pt') as f:
    for k in f.keys():
      params[k] = f.get_tensor(k)
  return params

param = {}
state_dict = {}

for f in os.listdir(model_id):
  if f.endswith('safetensors'):
    load_to(f'{model_id}/{f}', state_dict)

def flood(w, device=None):
  if w is None:
    return w
  device = device or w.device
  if w.dtype == torch.float8_e4m3fn:
    ws, w = getattr(w, 'scale_inv', None), w.view(torch.uint8).to(device)
    w[w == 128] = 0
    if ws is not None:
      w.scale_inv = ws.to(device)
  return w.to(device)

def load_tensor_fp8(key, device='cuda', fp8_to_bf16=False):
  w = state_dict[key].to(device)
  try:
    w.scale_inv = state_dict[key + '_scale_inv'].float().to(device)
  except:
    if w.dtype != torch.bfloat16:
      w = ops.from_float4_groupwise(w.to(device), state_dict[f'{key}_scale'].to(device), state_dict[f'{key}_scale_2'].to(device))
  if fp8_to_bf16 and w.dtype == torch.float8_e4m3fn:
    assert w.is_cuda
    w = flood(w)
    w = ops.to_bfloat16(w, w.scale_inv)
  return w

def world_slice(t, dim=0):
  if t is None:
    return t
  assert t.size(dim) % world_size == 0, f'Failed during slicing shape {list(t.shape)} on dimension {dim}.'
  group_size = t.size(dim) // world_size
  out = t.narrow(dim, world_rank * group_size, group_size).contiguous()
  if hasattr(t, 'scale_inv'):
    assert t.scale_inv.size(dim) % world_size == 0
    group_size = t.scale_inv.size(dim) // world_size
    out.scale_inv = t.scale_inv.narrow(dim, world_rank * group_size, group_size).contiguous()
  return out

master_print(f'Loading shared weights - 0.0% ..')

if model_type in ('deepseek_v3', 'kimi_k2'):
 for k in state_dict:
  if k.startswith('model.layers.'):
    if int(k.split('.')[2]) >= n_layers:
      continue
  if 'lm_head.weight' in k:
    param[k] = state_dict[k].to(torch.bfloat16).to(device)
    param[k].scale_inv = param[k]
    if param[k].dtype == torch.bfloat16:
      param[k], fp8scal = ops.to_float8_block(param[k])
      param[k].scale_inv = fp8scal
    continue
  if 'model.embed_tokens.weight' in k or 'norm' in k:
    param[k] = state_dict[k].to(torch.bfloat16).to(device)
    continue
  if '.rotary_emb.inv_freq' in k:
    continue
  if 'down_proj.' in k or 'gate_proj.' in k or 'up_proj.' in k:
    continue
  if '_exps' in k or k.endswith('_scale_inv') or 'kv_a_proj_with_mqa' in k:
    continue
  if 'self_attn.kv_b_proj.' in k or 'self_attn.q_b_proj.' in k:
    param[k] = world_slice(load_tensor_fp8(k, fp8_to_bf16=True), dim=0)
    continue
  if 'self_attn.o_proj.' in k:
    param[k] = flood(world_slice(load_tensor_fp8(k, 'cpu'), dim=1), device)
    if param[k].dtype == torch.bfloat16:
      param[k], fp8scal = ops.to_float8_block(param[k])
      param[k].scale_inv = fp8scal
    continue
  if 'self_attn.q_a_proj.weight' in k:
    Q = load_tensor_fp8(k, 'cpu')
    KV = load_tensor_fp8(k.replace('q_a_proj', 'kv_a_proj_with_mqa'), 'cpu')
    qkv = k.replace('q_a_proj', 'qkv_a_proj')
    param[qkv] = flood(torch.cat([Q.view(torch.uint8), KV.view(torch.uint8)], dim=0).view(Q.dtype)).to(device)
    if hasattr(Q, 'scale_inv'):
      param[qkv].scale_inv = torch.cat([Q.scale_inv, KV.scale_inv], dim=0).to(device)
    if param[qkv].dtype == torch.bfloat16:
      param_dim_size = param[qkv].size(0)
      padded_w = torch.nn.functional.pad(param[qkv], (0, 0, 0, (param_dim_size + 127) // 128 * 128 - param_dim_size))
      param[qkv], fp8scal = ops.to_float8_block(padded_w)
      param[qkv] = param[qkv].narrow(0, 0, param_dim_size)
      param[qkv].scale_inv = fp8scal
    continue
  if '.mlp.gate.e_score' in k or '.mlp.gate.weight' in k:
    param[k] = state_dict[k][:n_experts].contiguous().to(torch.bfloat16).to(device)
    continue
  master_print('>>>', k, state_dict[k].shape, state_dict[k].dtype, state_dict[k].view(-1)[:5]); exit(0)

elif model_type in ('qwen3_moe', 'qwen3'):

 for k in state_dict:
  if k.startswith('model.layers.'):
    if int(k.split('.')[2]) >= n_layers:
      continue
  if 'norm' in k:
    param[k] = state_dict[k].to(torch.bfloat16).to(device)
    continue
  if 'down_proj.' in k or 'gate_proj.' in k or 'up_proj.' in k or k.endswith('_scale_inv') or k.endswith('_scale') or k.endswith('_scale_2'):
    continue
  if '.mlp.gate.weight' in k:
    param[k] = state_dict[k][:n_experts].contiguous().to(torch.bfloat16).to(device)
    continue
  if 'self_attn.o_proj.' in k:
    param[k] = flood(world_slice(load_tensor_fp8(k, 'cpu'), dim=1), device)
    continue
  if 'lm_head.weight' in k or 'model.embed_tokens.weight' in k:
    param[k] = state_dict[k].contiguous().to(torch.bfloat16).to(device)
    continue
  if 'attn.k_' in k or 'attn.v_' in k:
    continue
  if 'attn.q_' in k:
    q_param = world_slice(load_tensor_fp8(k, fp8_to_bf16=True))
    k_param = world_slice(load_tensor_fp8(k.replace('.q_', '.k_'), fp8_to_bf16=True).view(num_key_value_heads, 1, head_dim, -1).repeat(1, max(1, world_size // num_key_value_heads), 1, 1).flatten(0, 2))
    v_param = world_slice(load_tensor_fp8(k.replace('.q_', '.v_'), fp8_to_bf16=True).view(num_key_value_heads, 1, head_dim, -1).repeat(1, max(1, world_size // num_key_value_heads), 1, 1).flatten(0, 2))
    param[k.replace('.q_', '.qkv_')] = torch.cat([q_param, k_param, v_param], dim=0)
    continue
  master_print('>>>', k, state_dict[k].shape, state_dict[k].dtype, state_dict[k].view(-1)[:5]); exit(0)

else:
  raise Exception(f'Unrecognized model type: {model_type}')

token_emb = param['model.embed_tokens.weight']
gate_moe = [param.get(f'model.layers.{i}.mlp.gate.weight', None) for i in range(n_layers)]

if model_type in ('deepseek_v3', 'kimi_k2'):
  lm_head = param['lm_head.weight']
  rms_ffn_w = [param[f'model.layers.{i}.post_attention_layernorm.weight'] for i in range(n_layers)]
  o_proj = [param[f'model.layers.{i}.self_attn.o_proj.weight'] for i in range(n_layers)]
  rms_att_w = [param[f'model.layers.{i}.input_layernorm.weight'] for i in range(n_layers)] + [param['model.norm.weight']]
  q_a_norm = [param[f'model.layers.{i}.self_attn.q_a_layernorm.weight'] for i in range(n_layers)]
  kv_a_norm = [param[f'model.layers.{i}.self_attn.kv_a_layernorm.weight'] for i in range(n_layers)]
  qkv_a_proj = [param[f'model.layers.{i}.self_attn.qkv_a_proj.weight'] for i in range(n_layers)]
  q_b_proj = [param[f'model.layers.{i}.self_attn.q_b_proj.weight'] for i in range(n_layers)]
  kv_b_proj = [param[f'model.layers.{i}.self_attn.kv_b_proj.weight'] for i in range(n_layers)]
  gate_bias = [param.get(f'model.layers.{i}.mlp.gate.e_score_correction_bias', None) for i in range(n_layers)]
else:
  pass


def load_expert_weight(prefs, dim=None, dev='cpu'):
  if type(prefs) not in (tuple, list):
    prefs = [prefs]
  ws, ss, mi, mw = [], [], [], []

  def pad_at_dim(x, dim, new_size):
    padded_shape = list(x.shape)
    padded_shape[dim] = new_size
    y = torch.empty(padded_shape, dtype=x.dtype, device=x.device)
    y.narrow(dim, 0, x.size(dim)).copy_(x)
    y.narrow(dim, x.size(dim), new_size - x.size(dim)).zero_()
    return y

  use_fp4 = (not args.disable_fp4)

  for pref in prefs:
    if f'{pref}.weight_scale_inv' in state_dict:
      w, s = state_dict[f'{pref}.weight'], state_dict[f'{pref}.weight_scale_inv']
      if dim is not None:
        assert w.dtype == torch.float8_e4m3fn
        if s.size(dim) % world_size != 0:
          block_size = w.size(dim) // s.size(dim)
          s = pad_at_dim(s, dim, (s.size(dim) + world_size - 1) // world_size * world_size)
          w = pad_at_dim(w, dim, s.size(dim) * block_size)
        w, s = flood(world_slice(w.view(torch.uint8), dim=dim).view(w.dtype)), world_slice(s, dim=dim).float()
      if use_fp4:
        wd = w.to(device, non_blocking=True)
        sd = s.to(device, non_blocking=True)
        w, s, o = ops.to_float4_groupwise(ops.to_bfloat16(wd, sd))
        ws += [w]
        ss += [s]
        mi += [o.view(1)]
        mw += [o.view(1)]
        torch.cuda.synchronize()
        del w, s, o, wd, sd, state_dict[f'{pref}.weight'], state_dict[f'{pref}.weight_scale_inv']
        continue
      ws += [w]
      ss += [s]
      del state_dict[f'{pref}.weight'], state_dict[f'{pref}.weight_scale_inv']
      continue
    w, s = state_dict[f'{pref}.weight'], state_dict.get(f'{pref}.weight_scale', None)
    if dim is not None:
      w, s = world_slice(w, dim=dim), flood(world_slice(s, dim=dim))
    if w.dtype == torch.bfloat16 and use_fp4 and len([_ for _ in gate_moe if _ is not None]) > 0:
      w, s, o = ops.to_float4_groupwise(w)
      ws += [w]
      ss += [s]
      mi += [o.view(1)]
      mw += [o.view(1)]
      torch.cuda.synchronize()
      del w, s, o, state_dict[f'{pref}.weight']
      continue
    ws += [w]
    if s is None:
      continue 
    ss += [s]
    mi += [state_dict[f'{pref}.input_scale'].view(1)]
    mw += [state_dict[f'{pref}.weight_scale_2'].view(1)]
  if not ss:
    ws = torch.cat(ws, dim=0).unsqueeze(0).to(dev)
  elif not mw:
    ws = torch.cat(ws, dim=0).unsqueeze(0).to(dev).view(torch.float8_e4m3fn)
    ws.scale_inv = torch.cat(ss, dim=0).unsqueeze(0).to(dev)
  else:
    ws = torch.cat(ws, dim=0).unsqueeze(0).to(dev)
    ws.scale_inv = torch.cat(ss, dim=0).unsqueeze(0).to(dev).view(torch.float8_e4m3fn)
    ws.meta_input = torch.cat(mi, dim=0).unsqueeze(0).to(dev)
    ws.meta_weight = torch.cat(mw, dim=0).unsqueeze(0).to(dev)
  return ws


def load_experts():
  gate_up_p, down_p = [], []

  for i in range(n_layers):
    if gate_moe[i] is None:
      gate_up_proj = load_expert_weight([f'model.layers.{i}.mlp.gate_proj', f'model.layers.{i}.mlp.up_proj'], dim=-2, dev=device)
      down_proj = load_expert_weight(f'model.layers.{i}.mlp.down_proj', dim=-1, dev=device)
      gate_up_p += [gate_up_proj]
      down_p += [down_proj]
      continue
    master_print(f'Loading expert weights - {(i + 1) / n_layers * 100:.1f}% ..')

    def pack(proj, device):
      if proj[0].dtype == torch.bfloat16:
        local = ops.copy_to_device(proj)
      elif not hasattr(proj[0], 'meta_weight'):
        local = ops.copy_to_device(proj).view(torch.float8_e4m3fn)
        local.scale_inv = ops.copy_to_device([_.scale_inv for _ in proj])
      else:
        local = ops.copy_to_device(proj)
        local.scale_inv = ops.copy_to_device([_.scale_inv for _ in proj]).view(torch.float8_e4m3fn)
        local.meta_input = torch.cat([_.meta_input.view(1, -1) for _ in proj], dim=0).to(device)
        local.meta_weight = torch.cat([_.meta_weight.view(1, -1) for _ in proj], dim=0).to(device)
      return local

    gate_up_proj = [load_expert_weight([f'model.layers.{i}.mlp.experts.{ID}.gate_proj', f'model.layers.{i}.mlp.experts.{ID}.up_proj'], dim=-2) for ID in range(n_experts)]
    down_proj = [load_expert_weight(f'model.layers.{i}.mlp.experts.{ID}.down_proj', dim=-1) for ID in range(n_experts)]
    try:
      gate_up_proj += [load_expert_weight([f'model.layers.{i}.mlp.shared_experts.gate_proj', f'model.layers.{i}.mlp.shared_experts.up_proj'], dim=-2)]
      down_proj += [load_expert_weight(f'model.layers.{i}.mlp.shared_experts.down_proj', dim=-1)]
    except:
      pass
    local_device = device # if world_size > 1 else 'cpu'
    gate_up_p += [pack(gate_up_proj, device=local_device)]
    down_p += [pack(down_proj, device=local_device)]
    del gate_up_proj, down_proj

  return gate_up_p, down_p

gate_up_p, down_p = load_experts()

del state_dict

batch = int(os.environ.get('BATCH', 1))
args.max_seq_len = (args.max_seq_len + args.buffer_size - 1) // args.buffer_size * args.buffer_size
assert args.max_seq_len % args.buffer_size == 0
buffer_data = torch.empty([args.buffer_size], dtype=torch.int64, device=device)

token_emb = token_emb.view(token_emb.size(0), -1)

master_print('Syncing with other peers..')
peer_barrier()

shared_exp_id = torch.zeros([batch, 1], dtype=torch.int32, device=device)
shared_weights = torch.ones([batch, 1], dtype=torch.float32, device=device)
topk_exp_id = torch.zeros([batch, n_top_k + 1], dtype=torch.int32, device=device)
score_weight = torch.zeros([batch, n_top_k + 1], dtype=torch.float32, device=device)
score_weight[:, n_top_k], topk_exp_id[:, n_top_k] = 1.0, n_experts


if world_size > 1:
  sigp = torch.ops.tutel_ops.uncached_empty([8192 * 16], torch.int32)
  sigp = torch.ops.tutel_ops.uncached_exchange(sigp[0], net.simple_all_gather(sigp[1]), world_rank)
  buffer = torch.ops.tutel_ops.uncached_empty([batch, token_emb.size(-1)], torch.bfloat16)
  buffer = torch.ops.tutel_ops.uncached_exchange(buffer[0], net.simple_all_gather(buffer[1]), world_rank)
  torch.ops.tutel_ops.configure_buffers(sigp, buffer)

if model_type in ('deepseek_v3', 'kimi_k2'):
  if model_type == 'kimi_k2':
    cos_sin = fetch_url(f'/tmp/kimi_cos_sin.npy', 'https://huggingface.co/datasets/ghostplant/data-collections/resolve/main/kimi_k2_128K.npy')
  else:
    cos_sin = torch.from_numpy(np.load(os.environ['COS_SIN_PATH']))
  cos_sin = cos_sin.to(torch.float32).view(2, -1, 2, 32).permute(1, 0, 3, 2).contiguous().view(-1, 2, 64).view(torch.int64).to(device)

  softmax_scale = (1 + 0.1 * math.log(config['rope_scaling']['factor'])) ** 2 * (config['qk_nope_head_dim'] + config['qk_rope_head_dim']) ** -0.5
  torch.ops.tutel_ops.deepseek_r1_prepare_weights(n_heads // world_size, args.max_seq_len, batch, softmax_scale,
      token_emb, lm_head, getattr(lm_head, 'scale_inv', lm_head), cos_sin, shared_exp_id, shared_weights, topk_exp_id, score_weight,
      rms_att_w, rms_ffn_w, qkv_a_proj, [getattr(qkv_a_proj[l], 'scale_inv', qkv_a_proj[l]) for l in range(n_layers)], q_a_norm, kv_a_norm, q_b_proj,
      kv_b_proj, o_proj, [getattr(o_proj[l], 'scale_inv', o_proj[l]) for l in range(n_layers)], [x for x in gate_moe if x is not None], [x for x in gate_bias if x is not None],
      gate_up_p, down_p, [_.scale_inv for _ in gate_up_p], [_.scale_inv for _ in down_p],
      [_.meta_input for _ in gate_up_p if hasattr(_, 'meta_input')], [_.meta_input for _ in down_p if hasattr(_, 'meta_input')],
      [_.meta_weight for _ in gate_up_p if hasattr(_, 'meta_weight')], [_.meta_weight for _ in down_p if hasattr(_, 'meta_weight')],
  )

  forward_fn = torch.ops.tutel_ops.deepseek_r1_forward
  use_cugraph = gate_up_p[-1].is_cuda
elif model_type in ('qwen3_moe', 'qwen3'):
  module = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./models/qwen3_moe.py")
  exec(compile(open(module).read(), filename=module, mode='exec'))
  use_cugraph = True
else:
  raise Exception(f'Unrecognized model type: {model_type}')


logits = torch.zeros([batch, 1, token_emb.size(0)], dtype=torch.bfloat16, device=device)
token_in = torch.ones([batch, 1], dtype=torch.int64, device=device)
token_range = torch.tensor([0, 1], dtype=torch.int32, device=device)

replay = lambda position_id: forward_fn(token_in, token_range, logits, position_id)

if use_cugraph:
  g = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]

  for k, v in [(0, 256), (1, 4096), (2, args.max_seq_len)]:
    for i in range(3):
      replay(v - 1)
    torch.cuda.synchronize()
    with torch.cuda.graph(g[k]):
      replay(v - 1)
    torch.cuda.synchronize()

  replay = lambda pos: g[0 if pos <= 256 else (1 if pos <= 4096 else 2)].replay()

def forward(token, pos):
  token_range[1] = pos + 1
  token_in.copy_(token)
  replay(pos)
  return logits

def benchmark(seq_len=256, use_ones=False):
  n_steps = 100
  with torch.no_grad():
    position_id = seq_len - 1
    token = torch.randint(1, 2 if use_ones else 4096, [batch, 1], dtype=torch.int64, device=device)
    forward(token, position_id)

    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for _ in range(n_steps):
      replay(position_id)
    torch.cuda.synchronize()
    t_stop = time.perf_counter()
    master_print('>> Decode TPS Benchmark (bsz = %d, output_ctx = %d, n_gpus = %d): %.4f tokens/sec' % (batch, seq_len, world_size, token.numel() * n_steps / (t_stop - t_start),))

    prof_type = int(os.environ.get('PROF', 0))
    if prof_type:
      import autort
      # autort.perform(lambda: forward(token, position_id), n_steps)
      autort.perform(lambda: replay(position_id) or logits, n_steps)
      exit(0)


def display(results):
  if world_rank != 0:
    return
  decoded_word = tokenizer.decode(results)
  sys.stdout.write(decoded_word)
  return decoded_word

def generate(prompt_tokens, max_tokens=None, temperature=0):
  max_tokens = max_tokens or args.max_seq_len
  temperature = temperature / 1000.0

  response_buffers = []
  progress_mask = ['\r-', '\r\\', '\r|', '\r/']
  token_offset = 0
  torch.cuda.synchronize()
  t_start = time.perf_counter()

  with torch.no_grad():
    pos, token = 0, prompt_tokens[0].view(1, 1).repeat(batch, 1).contiguous()

    master_print()
    while pos < max_tokens:
      for _ in range(args.buffer_size):
        logits = forward(token, pos + _)
        if pos + _ + 1 < len(prompt_tokens):
          next_token = prompt_tokens[pos + _ + 1].view(1, 1).repeat(batch, 1)
        else:
          if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1).view(-1, 1)
          else:
            b, s = logits.size(0), logits.size(1)
            next_token = torch.multinomial(torch.softmax(logits.view(b * s, -1) / temperature, dim=1), num_samples=b * s).view(b, s)
        token = next_token.contiguous()
        buffer_data[_] = token[-1]

      spill_tokens = buffer_data.cpu().tolist() if args.buffer_size > 1 else [buffer_data.item()]
      try:
        _ = spill_tokens.index(eos_token_id, max(len(prompt_tokens) + 1 - pos, 0))
        if pos + _ <= len(prompt_tokens):
          raise ValueError
      except ValueError:
        _ = args.buffer_size
      if world_rank == 0:
        response_buffers.append(display(spill_tokens[:_]))
      pos += _
      if _ < args.buffer_size:
        break

  t_stop = time.perf_counter()
  master_print()
  return ''.join(response_buffers), (t_stop - t_start), pos + 1


def token_encode(user_prompt, system_prompt=None):
  text = tokenizer.apply_chat_template(
      [{"role": "user", "content": user_prompt}] + ([] if system_prompt is None else [{"role": "system", "content": system_prompt}]),
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=not args.disable_thinking
  )
  return tokenizer([text], return_tensors="pt")['input_ids'].view(-1) 


from http.server import BaseHTTPRequestHandler, HTTPServer

prompt_cnt = torch.empty([3], dtype=torch.int64)
tokens_buf = torch.empty([args.max_seq_len], dtype=torch.int64)

class S(BaseHTTPRequestHandler):
  def do_POST(self):
    if '/chat' in self.path:
      try:
        is_legacy = False
        prompt_messages = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        prompt_system = None
        if 'messages' in prompt_messages:
          for roles in prompt_messages['messages']:
            if roles['role'] == 'user':
              prompt_user = roles['content']
            if roles['role'] == 'system':
              prompt_system = roles['content']
        else:
          is_legacy, prompt_user = True, prompt_messages['text']

        tokens = token_encode(prompt_user.strip(), system_prompt=prompt_system)
        max_tokens = int(prompt_messages.get('max_tokens', args.max_seq_len))
        temperature = int(1000 * abs(float(prompt_messages.get('temperature', 0))))
        assert max_tokens <= args.max_seq_len, f"The custom max_tokens({max_tokens}) cannot exceed the args.max_seq_len({args.max_seq_len}) argument set at startup."

        prompt_cnt[0] = tokens.numel()
        prompt_cnt[1] = max_tokens
        prompt_cnt[2] = temperature
        net.simple_broadcast(prompt_cnt)
        net.simple_broadcast(tokens)
        response, time_cost, num_tokens = generate(tokens.to(device), max_tokens=max_tokens, temperature=temperature)
        self.log_message(f'Complete {num_tokens} tokens in {time_cost:.3f} sec (Decode TPS = {num_tokens / time_cost:.3f})')

        response_json = {"id": "main", "choices": [{"index": 0, "finish_reason": "stop", "message": { "role": "assistant", "content": response }}]}
      except Exception as ex:
        response = str(ex)
        response_json = {"error": str(ex)}

      self.send_response(200)
      self.send_header('Content-type', 'text/plain' if is_legacy else 'application/json')
      self.end_headers()
      if is_legacy:
        self.wfile.write(b'- ' + ''.join(response).encode('utf-8') + b'\n')
      else:
        self.wfile.write(json.dumps(response_json).encode('utf-8'))
      self.wfile.write(b'\n')
    else:
      self.send_response(200)
      self.send_header('Content-type', 'text/plain')
      self.end_headers()
      self.wfile.write("Unrecognized POST request for {}\n".format(self.path).encode('utf-8'))


def serve(server_class=HTTPServer, handler_class=S, port=args.listen_port):
  if world_rank == 0:
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    try:
      master_print(f'''
Start listening on :{args.listen_port}. Request examples:

  >> curl -X POST http://0.0.0.0:{args.listen_port}/chat -d '{{"text": "Given f(f(x)) = x * x - 11x + 36, calculate f(5)."}}'
  >> curl -X POST http://0.0.0.0:{args.listen_port}/chat -d '{{"messages": [{{"role": "user", "content": "Given f(f(x)) = x * x - 11x + 36, calculate f(5)."}}], "max_tokens": 1024 }}'
''')
      httpd.serve_forever()
    except KeyboardInterrupt:
      pass
    httpd.server_close()
  else:
    with torch.no_grad():
      while True:
        net.simple_broadcast(prompt_cnt)
        prompt_size, max_tokens, temperature = prompt_cnt.cpu().tolist()
        tokens = tokens_buf.narrow(0, 0, prompt_size)
        net.simple_broadcast(tokens)
        generate(tokens.to(device), max_tokens=max_tokens, temperature=temperature)

if __name__ == '__main__':
  user_prompt = args.prompt.strip()
  master_print()
  if user_prompt:
    generate(token_encode(user_prompt).to(device))
    master_print()
  for _ in [1, 64, 256, 4096] + ([args.max_seq_len] if args.max_seq_len > 4096 else []):
    if _ <= args.max_seq_len:
      benchmark(_)
  if args.serve:
    serve()

