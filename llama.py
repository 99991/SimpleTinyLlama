import torch
import torch.nn.functional as F

def llama(token_ids, d):
    # Look up embeddings for each token id
    x = d["model.embed_tokens.weight"][token_ids]

    for i in range(22):
        # Normalize, attention, normalize again, multi-layer perceptron
        x_norm = rmsnorm(x, d[f"model.layers.{i}.input_layernorm.weight"])

        x = x + attention(x_norm, i, d)

        x_norm = rmsnorm(x, d[f"model.layers.{i}.post_attention_layernorm.weight"])

        x = x + mlp(x_norm, i, d)

    # Normalize and final projection head
    x = rmsnorm(x, d["model.norm.weight"])
    
    logits = x @ d["lm_head.weight"].T

    return logits

def mlp(x, i, d):
    """Multi-layer perceptron."""
    gate_proj = d[f"model.layers.{i}.mlp.gate_proj.weight"]
    up_proj = d[f"model.layers.{i}.mlp.up_proj.weight"]
    down_proj = d[f"model.layers.{i}.mlp.down_proj.weight"]

    x = F.silu(x @ gate_proj.T) * (x @ up_proj.T)
    x = x @ down_proj.T

    return x

def rmsnorm(x, weight, eps=1e-5):
    """Root mean square layer normalization."""
    dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = weight * x
    return x.to(dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def attention(x, i, d):
    """Multi-head self-attention."""
    # For a detailed explanation of attention, see this video by Andrej Karpathy:
    # https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1h2m
    bsz, q_len, _ = x.shape
    dtype = x.dtype
    num_heads = 32
    head_dim = 64
    num_key_value_heads = 4
    hidden_size = 2048
    num_key_value_groups = 8

    q_proj = d[f"model.layers.{i}.self_attn.q_proj.weight"]
    k_proj = d[f"model.layers.{i}.self_attn.k_proj.weight"]
    v_proj = d[f"model.layers.{i}.self_attn.v_proj.weight"]
    o_proj = d[f"model.layers.{i}.self_attn.o_proj.weight"]

    query_states = (x @ q_proj.T).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = (x @ k_proj.T).view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = (x @ v_proj.T).view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]

    # Compute rotary embedding
    base = 10000
    dim = 64

    t = torch.arange(kv_seq_len, device=x.device)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().view(1, 1, kv_seq_len, dim)
    sin = emb.sin().view(1, 1, kv_seq_len, dim)

    position_ids = torch.arange(q_len, device=x.device).view(1, q_len)

    # Apply rotary embedding
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    query_states = (query_states * cos) + (rotate_half(query_states) * sin)
    key_states = (key_states * cos) + (rotate_half(key_states) * sin)

    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * head_dim**-0.5

    # Create attention mask matrix like:
    # [0, -inf, -inf, -inf]
    # [0,    0, -inf, -inf]
    # [0,    0,    0, -inf]
    # [0,    0,    0,    0]
    attention_mask = torch.triu(torch.full((q_len, kv_seq_len), -float("inf"), device=x.device), 1).reshape(1, 1, q_len, kv_seq_len)

    assert attn_weights.size() == (bsz, num_heads, q_len, kv_seq_len)

    # To predict future tokens, only previous tokens may be used.
    # This is ensured by weighting future tokens very negatively,
    # so they are not chosen by the softmax.
    attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, hidden_size)

    attn_output = attn_output @ o_proj.T

    return attn_output
