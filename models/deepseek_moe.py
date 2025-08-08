# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION
from tutel import ops


def modeling(is_training=False):
  if model_type == 'kimi_k2':
    cos_sin = torch.from_numpy(np.load(os.environ['K2_COS_SIN_PATH']))
  else:
    cos_sin = torch.from_numpy(np.load(os.environ['R1_COS_SIN_PATH']))
  cos_sin = cos_sin.to(torch.float32).view(2, -1, 2, 32).permute(1, 0, 3, 2).contiguous().view(-1, 2, 64).view(torch.int64).to(device)

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

  softmax_scale = (1 + 0.1 * math.log(config['rope_scaling']['factor'])) ** 2 * (config['qk_nope_head_dim'] + config['qk_rope_head_dim']) ** -0.5
  n_local_heads = n_heads // world_size

  kv_cache = torch.randn([n_layers, args.max_seq_len, batch, 512 + 64], dtype=token_emb.dtype, device=token_emb.device)

  if model_type == 'kimi_k2':
    sigmoid_scaled_routing = torch.ops.tutel_ops.kimi_moe_sigmoid_scaled_topk
  else:
    sigmoid_scaled_routing = torch.ops.tutel_ops.deepseek_moe_sigmoid_scaled_topk

  shared_exp_id = torch.zeros([batch, 1], dtype=torch.int32, device=device)
  shared_weights = torch.ones([batch, 1], dtype=torch.float32, device=device)
  topk_exp_id = torch.zeros([batch, n_top_k + 1], dtype=torch.int32, device=device)
  score_weight = torch.zeros([batch, n_top_k + 1], dtype=torch.float32, device=device)
  score_weight[:, n_top_k], topk_exp_id[:, n_top_k] = 1.0, n_experts

  def forward_fn(token_in, token_range, logits, max_pos):
    x = token_in
    samples = x.numel()
    assert x.is_cuda and x.ndim == 2
    assert topk_exp_id.ndim == 2

    x = token_emb.index_select(0, x.flatten()).view(x.size(0), x.size(1), token_emb.size(1))
    xb = torch.ops.tutel_ops.rmsnorm_bf16(x, rms_att_w[0], 1e-6, 0)

    for l in range(n_layers):
      xb = torch.ops.tutel_ops.deepseek_r1_attn_bf16xf8_block_scal_v2(xb, kv_cache[l], cos_sin,
         qkv_a_proj[l], getattr(qkv_a_proj[l], 'scale_inv', qkv_a_proj[l]),
         q_a_norm[l], kv_a_norm[l], q_b_proj[l], kv_b_proj[l],
         o_proj[l], getattr(o_proj[l], 'scale_inv', o_proj[l]),
         token_range, buffer[0], max_pos, n_local_heads, softmax_scale)

      x = torch.ops.tutel_ops.intra_add_allreduce_bf16(x, xb, sigp, buffer, False)
      xb = torch.ops.tutel_ops.rmsnorm_bf16(x, rms_ffn_w[l], 1e-6, 0)

      if gate_moe[l] is None:
        topk_ids, topk_weights = shared_exp_id, shared_weights
      else:
        logits_bf16 = torch.ops.tutel_ops.gate_gemm_out_bf16(xb, gate_moe[l])
        sigmoid_scaled_routing(logits_bf16, gate_bias[l], score_weight, topk_exp_id)
        topk_ids, topk_weights = topk_exp_id, score_weight

      if l >= len(gate_up_p):
        pass
      elif not gate_up_p[l].is_cuda:
        assert topk_ids.size(1) > 1 and world_size == 1
        xb = torch.ops.tutel_ops.glu_expert_bf16_host_mm(xb, topk_ids, topk_weights,
            gate_up_p[l], down_p[l], l)
      elif hasattr(gate_up_p[l], 'meta_weight'):
        xb = torch.ops.tutel_ops.glu_expert_bf16xf4_group_scal(xb, topk_ids, topk_weights,
            gate_up_p[l], gate_up_p[l].scale_inv, gate_up_p[l].meta_input, gate_up_p[l].meta_weight,
            down_p[l], down_p[l].scale_inv, down_p[l].meta_input, down_p[l].meta_weight, buffer[0])
      else:
        xb = torch.ops.tutel_ops.glu_expert_bf16xf8_block_scal(xb, topk_ids, topk_weights,
            gate_up_p[l], gate_up_p[l].scale_inv, down_p[l], down_p[l].scale_inv, buffer[0])

      x = torch.ops.tutel_ops.intra_add_allreduce_bf16(x, xb, sigp, buffer, samples > 1)
      xb = torch.ops.tutel_ops.rmsnorm_bf16(x, rms_att_w[l + 1], 1e-6, 0)

    torch.ops.tutel_ops.gemm_nt_bf16xfp8_block_scal_out(xb, lm_head, getattr(lm_head, 'scale_inv', lm_head), logits)
    return logits

  return forward_fn

forward_fn = modeling()
use_cugraph = True
