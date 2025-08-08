# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION
from tutel import ops
import numpy as np
import autort

import torch._dynamo.config
import torch._inductor.config

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True


def modeling(is_training=False):

  inv_freq = torch.from_numpy(np.load(os.environ['GPT_FREQ_PATH'])).view(-1, 1)
  rope_scaling_factor = 1.3465735902799727
  emb = (inv_freq @ torch.arange(0, args.max_seq_len, dtype=torch.float32).view(1, -1)).t().reshape(1, args.max_seq_len, -1)
  cos_sin = torch.cat([emb.cos() * rope_scaling_factor, emb.sin() * rope_scaling_factor], dim=0).bfloat16().to(device)

  gate_bias = [param[f'model.layers.{i}.mlp.router.bias'] for i in range(n_layers)]
  sinks = [param[f'model.layers.{l}.self_attn.sinks'].to(device) for l in range(n_layers)]

  lm_head = param['lm_head.weight']
  rms_end_w = param['model.norm.weight']
  rms_att_w = [param[f'model.layers.{i}.input_layernorm.weight'] for i in range(n_layers)] + [rms_end_w]
  rms_ffn_w = [param[f'model.layers.{i}.post_attention_layernorm.weight'] for i in range(n_layers)]
  qkv_proj = [param[f'model.layers.{l}.self_attn.qkv_proj.weight'].to(device) for l in range(n_layers)]
  o_proj = [param[f'model.layers.{i}.self_attn.o_proj.weight'] for i in range(n_layers)]
  sm_scale = 0.125
  sliding_window_size = 128
  padding_size = 8

  key_cache = torch.zeros([n_layers, batch, args.max_seq_len + padding_size, max(1, num_key_value_heads // world_size), head_dim], dtype=torch.bfloat16, device=device)
  val_cache = torch.zeros([n_layers, batch, args.max_seq_len + padding_size, max(1, num_key_value_heads // world_size), head_dim], dtype=torch.bfloat16, device=device)

  # FIXME: Initial Non-flash and Non-SWA version
  def sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    token_range,
    scaling,
    sinks,
    sliding_window_size,
  ):
    if sliding_window_size is None:
      sliding_window_size = args.max_seq_len

    max_length = k.size(1)
    num_key_value_groups = q.size(2) // k.size(2)

    qk = torch.einsum('bhHm,bshm->bhHs', [q.view(q.size(0) * q.size(1), k.size(2), num_key_value_groups, q.size(-1)), k]).view(-1, max_length)
    qk_out = torch.empty([qk.size(0), qk.size(1) + padding_size], dtype=qk.dtype, device=qk.device)

    autort.ops.scaled_mask_sliding_sink_inv_bf16(qk, token_range, sinks.view(qk.size(0)), qk_out, extra=[scaling, sliding_window_size])
    qk_out = torch.softmax(qk_out, -1)
    o = torch.einsum('bhHs,bshm->bhHm', [qk_out.view(q.size(0) * q.size(1), k.size(2), num_key_value_groups, qk_out.size(-1)), v]).flatten(1, 2).unsqueeze(1)
    return o

  @torch.compile
  def rms_norm(hidden_states, weight, eps=None):
    eps = eps if eps is not None else 1e-5
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return (weight * hidden_states).to(input_dtype)

  @torch.compile
  def add_norm(x, xb, weight, eps=None):
    x = x + xb
    return x, rms_norm(x, weight, eps=eps)

  @torch.compile
  def activate(xb):
    gate, up = xb[..., ::2], xb[..., 1::2]
    clamp_limit, sigmoid_alpha = 7.0, 1.702
    gate = gate.clamp(min=None, max=clamp_limit)
    up = up.clamp(min=-clamp_limit, max=clamp_limit)
    glu = gate * torch.sigmoid(gate * sigmoid_alpha)
    glu = (up + 1) * glu
    return glu

  @torch.compile
  def combine(xb, selected_weights, samples):
    return (xb.view(samples, 4, -1) * selected_weights.view(samples, -1, 1)).sum(dim=1)

  @torch.compile
  def gating(router_logits):
    router_top_value, router_indices = torch.topk(router_logits, 4, dim=-1)
    router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
    return router_indices.to(torch.int32), router_top_value

  @torch.compile
  def apply_rotary_emb(
      qkv_out: torch.Tensor,
      key_cache,
      val_cache,
  ) -> torch.Tensor:
    q_heads, kv_heads = n_heads // world_size, max(1, num_key_value_heads // world_size)

    position_ids = token_range[1:] - token_range[:-1] - 1
    cos = cos_sin[0].index_select(0, position_ids).unsqueeze(1)
    sin = cos_sin[1].index_select(0, position_ids).unsqueeze(1)

    qk_states = qkv_out.narrow(-2, 0, q_heads + kv_heads)
    first_half, second_half = torch.chunk(qk_states, 2, dim=-1)
    first_ = first_half * cos - second_half * sin
    second_ = second_half * cos + first_half * sin
    qk_states = torch.cat((first_, second_), dim=-1)

    q_states = qk_states.narrow(-2, 0, q_heads)
    k_states = qk_states.narrow(-2, q_heads, kv_heads)
    v_states = qkv_out.narrow(-2, q_heads + kv_heads, kv_heads)

    key_cache.index_put_([position_ids], k_states[0])
    val_cache.index_put_([position_ids], v_states[0])
    val_cache.index_put_([position_ids + 1], torch.zeros_like(v_states[0]))
    return q_states

  @torch.compiler.disable(recursive=True)
  def gemm(x, w, bias):
    samples = x.numel() // x.size(-1)
    if samples == 1:
      return autort.ops.gemv_bf16(bias.flatten(), x.flatten().view(torch.int32), w.view(torch.int32)).view(*x.shape[:-1], w.size(0))
    else:
      return torch.addmm(bias.view(samples, -1), x.view(samples, x.size(-1)), w.t()).view(*x.shape[:-1], -1)

  def forward_fn(token_in, token_range, logits, max_pos):
    x = token_in
    samples = x.numel()
    assert x.is_cuda and x.ndim == 2
    max_pos = max(max_pos, sliding_window_size - 1)

    x = token_emb.index_select(0, x.flatten()).view(x.size(0), x.size(1), token_emb.size(1))
    xb = rms_norm(x, rms_att_w[0])

    for l in range(n_layers):
      qkv_out = gemm(xb, qkv_proj[l], qkv_proj[l].bias).view(xb.size(0), xb.size(1), -1, head_dim)

      q_states = apply_rotary_emb(qkv_out, key_cache[l, 0], val_cache[l, 0])

      scores = sliding_window_attention(q_states, key_cache[l, :, :max_pos + 1], val_cache[l, :, :max_pos + 1 + padding_size], token_range,
            scaling=sm_scale, sinks=sinks[l], sliding_window_size=sliding_window_size if l % 2 == 0 else None)

      xb = gemm(scores.flatten(-2), o_proj[l], o_proj[l].bias)

      # FIXME: Distributed mode not enabled
      if world_size > 1:
        x = torch.ops.tutel_ops.intra_add_allreduce_bf16(x, xb, sigp, buffer, True)
        xb = rms_norm(x, rms_ffn_w[l])
      else:
        x, xb = add_norm(x, xb, rms_ffn_w[l])

      router_logits = gemm(xb, gate_moe[l], gate_bias[l]).view(-1, gate_moe[l].size(0))
      selected_experts, selected_weights = gating(router_logits)

      # FIXME: To fine-tune MVFP4 for different GPUs
      xb = autort.ops.fmoe_f16xf4_mxfp_bmm(xb.view(torch.int32).view(-1, xb.size(-1) // 32, 16), gate_up_p[l].scales, selected_experts.flatten(), gate_up_p[l].flatten(2).view(torch.complex128), gate_up_p[l].bias, extra=[2])
      xb = activate(xb)
      xb = autort.ops.fmoe_f16xf4_mxfp_bmm(xb.view(torch.int32).view(-1, xb.size(-1) // 32, 16), down_p[l].scales, selected_experts.flatten(), down_p[l].flatten(2).view(torch.complex128), down_p[l].bias, extra=[0])
      xb = combine(xb, selected_weights, samples).view(x.shape)

      if world_size > 1:
        x = torch.ops.tutel_ops.intra_add_allreduce_bf16(x, xb, sigp, buffer, True)
        xb = rms_norm(x, rms_att_w[l + 1])
      else:
        x, xb = add_norm(x, xb, rms_att_w[l + 1])

    torch.matmul(xb, lm_head.t(), out=logits.view(xb.size(0), xb.size(1), lm_head.size(0)))

  return forward_fn

forward_fn = modeling()
use_cugraph = True

