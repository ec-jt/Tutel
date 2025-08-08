# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION
from tutel import ops

is_triton_supported = (IS_HIP_EXTENSION and torch.cuda.get_device_properties().gcnArchName.split(':')[0] >= 'gfx90a') or \
                      (not IS_HIP_EXTENSION and torch.cuda.get_device_capability()[0] >= 8)

import torch._dynamo.config
import torch._inductor.config

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

module = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./models/sglang_group_attn.py")
exec(compile(open(module).read(), filename=module, mode='exec'))

def modeling(is_training=False):
  inv_freq_expanded = 1 / (config['rope_theta'] ** (torch.arange(0, head_dim // 2) / (head_dim / 2.0))).view(1, -1, 1)
  position_ids_expanded = torch.arange(0, args.max_seq_len, dtype=torch.float32).view(1, 1, -1)
  freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
  emb = torch.cat((freqs, freqs), dim=-1)
  cos_sin = torch.cat([emb.cos(), emb.sin()], dim=0).bfloat16().to(device)

  rms_end_w = param['model.norm.weight']
  weight_classify = param['lm_head.weight']
  rms_ffn_w = [param[f'model.layers.{i}.post_attention_layernorm.weight'] for i in range(n_layers)]
  rms_att_w = [param[f'model.layers.{i}.input_layernorm.weight'] for i in range(n_layers)] + [rms_end_w]
  o_proj = [param[f'model.layers.{i}.self_attn.o_proj.weight'] for i in range(n_layers)]
  qkv_proj = [param[f'model.layers.{l}.self_attn.qkv_proj.weight'].to(device) for l in range(n_layers)]
  q_norm = [param[f'model.layers.{l}.self_attn.q_norm.weight'].to(device) for l in range(n_layers)]
  k_norm = [param[f'model.layers.{l}.self_attn.k_norm.weight'].to(device) for l in range(n_layers)]
  qk_norm = [torch.cat([q_norm[l].unsqueeze(0), k_norm[l].unsqueeze(0)]) for l in range(n_layers)]
  key_cache = torch.zeros([n_layers, batch, args.max_seq_len, max(1, num_key_value_heads // world_size), head_dim], dtype=torch.bfloat16, device=device)
  val_cache = torch.zeros([n_layers, batch, args.max_seq_len, max(1, num_key_value_heads // world_size), head_dim], dtype=torch.bfloat16, device=device)

  max_kv_splits = 32
  num_kv_splits = torch.full([batch + 1], max_kv_splits, dtype=torch.int32, device=device)
  kv_indices = torch.arange(0, args.max_seq_len + 1, dtype=torch.int32, device=device)
  INF = torch.tensor(float('-inf'), dtype=key_cache.dtype, device=device)

  def attn_decode_fwd(q, k_buffer, v_buffer, kv_indptr, max_pos, sm_scale=None):
    k_buffer, v_buffer = k_buffer.narrow(1, 0, max_pos), v_buffer.narrow(1, 0, max_pos)
    sm_scale = sm_scale or (1 / math.sqrt(head_dim))
    if not is_triton_supported:
      qk = torch.einsum('bhHm,bshm->bhHs', [q.view(q.size(0), k_buffer.size(2), -1, q.size(2)), k_buffer])
      if is_training:
        qk = torch.softmax(torch.where(torch.arange(0, qk.size(-1), dtype=kv_indptr.dtype, device=device).view(1, 1, 1, -1) < kv_indptr[1], qk * sm_scale, INF), -1)
      else:
        qk = torch.softmax(ops.scaled_mask_inv(qk, kv_indptr, sm_scale), -1)
      o = torch.einsum('bhHs,bshm->bhHm', [qk, v_buffer]).view(q.size())
      return o.unsqueeze(1)

    n_heads = q.size(-2)
    o = torch.empty([batch, n_heads, head_dim], dtype=torch.bfloat16, device=device)
    attn_logits = torch.empty([batch, n_heads, max_kv_splits, head_dim], dtype=torch.float32, device=device)
    attn_lse = torch.empty([batch, n_heads, max_kv_splits], dtype=torch.float32, device=device)

    decode_attention_fwd_grouped(q, k_buffer.flatten(0, 1),
      v_buffer.flatten(0, 1),
      o,
      kv_indptr,
      kv_indices,
      attn_logits,
      attn_lse,
      num_kv_splits,
      max_kv_splits,
      sm_scale,
      logit_cap=0.0)
    return o.unsqueeze(1)

  def rms_norm(x, weight, eps=1e-6):
    if not is_triton_supported:
        return ops.rmsnorm_bf16(x, weight, eps, 0) if not is_training else torch.nn.functional.rms_norm(x, weight.shape, weight, eps=eps)

    input_dtype = x.dtype
    x = x.float()
    variance = (x * x).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)

  def add_norm(x, xb, weight, eps=1e-6):
    x = x + xb
    return x, rms_norm(x, weight, eps)

  def silu_mul(xb):
    return torch.nn.functional.silu(xb.narrow(-1, 0, xb.size(-1) // 2)) * xb.narrow(-1, xb.size(-1) // 2, xb.size(-1) // 2)

  fn = torch.compile if is_triton_supported else (lambda fn, **kargs: fn)

  @torch.compiler.disable(recursive=True)
  def glu_ffn(xb, selected_experts, routing_weights, w13, w2):
    if w13.dtype == torch.float8_e4m3fn and w2.dtype == torch.float8_e4m3fn:
      xb = ops.glu_expert_bf16xf8_block_scal(xb, selected_experts, routing_weights, w13, w13.scale_inv, w2, w2.scale_inv, xb)
    elif w13.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16:
      assert selected_experts.size(-1) == 1, "Unhandled MoE datatype: topk > 1 for bfloat16"
      xb = torch.matmul(xb, w13[0].t())
      xb = silu_mul(xb) if not is_triton_supported else fn(silu_mul)(xb)
      xb = torch.matmul(xb, w2[0].t())
    elif hasattr(w13, 'meta_weight') and hasattr(w2, 'meta_weight'):
      xb = ops.glu_expert_bf16xf4_group_scal(xb, selected_experts, routing_weights, w13, w13.scale_inv, w13.meta_input, w13.meta_weight, w2, w2.scale_inv, w2.meta_input, w2.meta_weight, xb)
    else:
      raise Exception(f"Unhandled MoE datatype: {w13.dtype}, {w2.dtype}")
    return xb

  @torch.compiler.disable(recursive=True)
  def gemm(x, y, out=None):
    assert y.ndim == 2
    out = out if out is not None else torch.empty(list(x.shape[:-1]) + [y.size(0)], dtype=x.dtype, device=x.device)
    return ops.gemm_nt_bf16xfp8_block_scal_out(x, y, getattr(y, 'scale_inv', y), out)

  def local_compile(forward_fn):
    if world_size > 1:
      return fn(forward_fn)
    return fn(forward_fn, mode='max-autotune-no-cudagraphs')

  COS, SIN = cos_sin[0].view(-1, head_dim).bfloat16().view(torch.int32), cos_sin[1].view(-1, head_dim).bfloat16().view(torch.int32)
  local_kv_heads = max(1, num_key_value_heads // world_size)

  shared_topk_weights = torch.ones([batch, 1], dtype=torch.float32, device=device)
  shared_topk_ids = torch.zeros([batch, 1], dtype=torch.int32, device=device)

  def forward_fn(token_in, token_range, logits, max_pos):
    x = token_emb.index_select(0, token_in.flatten()).view(*token_in.shape, token_emb.size(-1))
    xb = rms_norm(x, rms_att_w[0])

    for l in range(n_layers):
      qkv_out = gemm(xb, qkv_proj[l]).view(x.size(0) * x.size(1), -1, head_dim)
      if not is_training:
        q_states = ops.qwen3_norm_rotary_kvcache2_bf16(COS, SIN, token_range, qkv_out, key_cache[l], val_cache[l], qk_norm[l], n_heads // world_size)
      else:
        q_heads, kv_heads = n_heads // world_size, key_cache[l].size(-2)
        q_states, k_states, v_states = rms_norm(qkv_out.narrow(-2, 0, q_heads), qk_norm[l][0]), rms_norm(qkv_out.narrow(-2, q_heads, kv_heads), qk_norm[l][1]), qkv_out.narrow(-2, q_heads + kv_heads, kv_heads)
        position_ids = token_range[1:] - token_range[:-1] - 1

        def apply_rotary_pos_emb(q, k, cos, sin):
            def rotate_half(x):
                x1 = x[..., : x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1)

            cos = cos.view(*q.shape[:-2], -1, q.size(-1))
            sin = sin.view(*q.shape[:-2], -1, q.size(-1))
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            return q_embed, k_embed

        q_states, k_states = apply_rotary_pos_emb(q_states, k_states, COS[position_ids].view(torch.bfloat16), SIN[position_ids].view(torch.bfloat16))
        key_cache[l][0].index_put_([position_ids], k_states)
        val_cache[l][0].index_put_([position_ids], v_states)

      scores = attn_decode_fwd(q_states, key_cache[l], val_cache[l], token_range, max_pos)
      xb = gemm(scores.flatten(2), o_proj[l])

      if world_size > 1:
        x = torch.ops.tutel_ops.intra_add_allreduce_bf16(x, xb, sigp, buffer, True)
        xb = rms_norm(x, rms_ffn_w[l])
      else:
        x, xb = add_norm(x, xb, rms_ffn_w[l])

      if gate_moe[l] is None:
        routing_weights, selected_experts = shared_topk_weights, shared_topk_ids
      else:
        gate_out = torch.nn.functional.softmax(torch.matmul(xb, gate_moe[l].t()), -1, dtype=torch.float32)
        routing_weights, selected_experts = ops.qwen3_moe_scaled_topk(gate_out)
      xb = glu_ffn(xb, selected_experts, routing_weights, gate_up_p[l], down_p[l])

      if world_size > 1:
        x = torch.ops.tutel_ops.intra_add_allreduce_bf16(x, xb, sigp, buffer, True)
        xb = rms_norm(x, rms_att_w[l + 1])
      else:
        x, xb = add_norm(x, xb, rms_att_w[l + 1])

    torch.matmul(xb, weight_classify.t(), out=logits)
  return local_compile(forward_fn)

forward_fn = modeling()
use_cugraph = True
