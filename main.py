from tokenizer import ChatTokenizer
import util
import torch
import torch.nn.functional as F

def main():
    # Choose computation device. You might have to change this to just "cpu"
    # if you do not have enough VRAM. For Apple M1/2 chips, "mps" might work.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    # Choose data type here. Your hardware might not support torch.bfloat16
    # or torch.float16, but torch.float32 should usually work. However,
    # it will require twice as much memory.
    dtype = None
    #dtype = torch.float32

    model_filename = "data/TinyLlama-1.1B-Chat-v1.0/model.safetensors"
    model_url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors?download=true"
    tokenizer_filename = "data/TinyLlama-1.1B-Chat-v1.0/tokenizer.model"
    tokenizer_url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.model?download=true"

    # Download tokenizer and model
    util.download(tokenizer_url, tokenizer_filename)
    util.download(model_url, model_filename)

    # Load the model weights
    state_dict = util.load_safetensors(model_filename, device, dtype)

    # Create the tokenizer
    tokenizer = ChatTokenizer(tokenizer_filename)

    # Question loop
    while True:
        # Ask for a question
        try:
            ask_question = "\x1b[33mAsk a question:\x1b[0m\n"
            prompt = input(ask_question)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        try:
            # Convert to chat prompt format
            chat_prompt = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"

            # Convert prompt to a list of token ids
            token_ids = tokenizer.encode(chat_prompt)

            # Positions of tokens to process
            position_ids = list(range(len(token_ids)))

            cache = {}

            # Generate tokens one by one
            while len(token_ids) < 2048:
                # Upload token ids and position ids to device
                token_ids_torch = torch.tensor([[token_ids[i] for i in position_ids]], device=device)
                position_ids = torch.tensor(position_ids, device=device).view(1, -1)

                # Predict logits
                logits = llama(token_ids_torch, position_ids, cache, state_dict)

                # Choose the most likely token
                new_token_id = logits[0, -1].argmax().item()

                # Stop if we reach the special end token, which indicates the end of the response
                if new_token_id == tokenizer.end_token_id:
                    break

                # Update position of tokens to process
                position_ids = [len(token_ids)]

                # Append the new token to the list of tokens
                token_ids.append(new_token_id)

                # Decode all tokens so far
                response = tokenizer.decode(token_ids)

                # Clear screen, print prompt and response (skipping prompt part)
                clear_screen = "\x1b[2J\x1b[H"
                green_color = "\x1b[32m"
                reset_color = "\x1b[0m\n"
                print(clear_screen + ask_question + prompt + "\n")
                print(green_color + response[len(chat_prompt):] + reset_color)
        except KeyboardInterrupt:
            print("Response interrupted")

def precompute(device):
    # Compute rotary embedding
    max_seq_len = 2048
    base = 10000
    dim = 64
    t = torch.arange(max_seq_len, device=device)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().view(max_seq_len, dim)
    sin = emb.sin().view(max_seq_len, dim)

    # Create attention mask matrix like:
    # [0, -inf, -inf, -inf]
    # [0,    0, -inf, -inf]
    # [0,    0,    0, -inf]
    # [0,    0,    0,    0]
    attention_mask = torch.triu(torch.full((max_seq_len, max_seq_len), -float("inf"), device=device), 1)

    return cos, sin, attention_mask

def llama(token_ids, position_ids, cache, state_dict):
    # Look up embeddings for each token id
    x = state_dict["model.embed_tokens.weight"][token_ids]

    for i in range(22):
        # Normalize, attention, normalize again, multi-layer perceptron
        x_norm = rmsnorm(x, state_dict[f"model.layers.{i}.input_layernorm.weight"])

        x = x + attention(x_norm, position_ids, i, cache, state_dict)

        x_norm = rmsnorm(x, state_dict[f"model.layers.{i}.post_attention_layernorm.weight"])

        x = x + mlp(x_norm, i, state_dict)

    # Normalize and final projection head
    x = rmsnorm(x, state_dict["model.norm.weight"])

    logits = x @ state_dict["lm_head.weight"].T

    return logits

def mlp(x, i, state_dict):
    """Multi-layer perceptron."""
    gate_proj = state_dict[f"model.layers.{i}.mlp.gate_proj.weight"]
    up_proj = state_dict[f"model.layers.{i}.mlp.up_proj.weight"]
    down_proj = state_dict[f"model.layers.{i}.mlp.down_proj.weight"]

    x = F.silu(x @ gate_proj.T) * (x @ up_proj.T)
    x = x @ down_proj.T

    return x

def rmsnorm(x, weight, eps=1e-5):
    """Root mean square layer normalization."""
    dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(dtype)

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

def attention(x, position_ids, i, cache, state_dict):
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

    if "precomputed" not in cache:
        cache["precomputed"] = precompute(x.device)

    cos, sin, attention_mask = cache["precomputed"]

    q_proj = state_dict[f"model.layers.{i}.self_attn.q_proj.weight"]
    k_proj = state_dict[f"model.layers.{i}.self_attn.k_proj.weight"]
    v_proj = state_dict[f"model.layers.{i}.self_attn.v_proj.weight"]
    o_proj = state_dict[f"model.layers.{i}.self_attn.o_proj.weight"]

    query_states = (x @ q_proj.T).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = (x @ k_proj.T).view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = (x @ v_proj.T).view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

    # Apply rotary embedding
    partial_cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    partial_sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    query_states = (query_states * partial_cos) + (rotate_half(query_states) * partial_sin)
    key_states = (key_states * partial_cos) + (rotate_half(key_states) * partial_sin)

    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    if f"kv_states_{i}" in cache:
        cached_key_states, cached_value_states = cache[f"kv_states_{i}"]

        key_states = torch.cat([cached_key_states, key_states], dim=2)
        value_states = torch.cat([cached_value_states, value_states], dim=2)

    cache[f"kv_states_{i}"] = (key_states, value_states)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * head_dim**-0.5

    # To predict future tokens, only previous tokens may be used.
    # This is ensured by weighting future tokens very negatively,
    # so they are not chosen by the softmax.
    attn_weights = attn_weights + attention_mask[position_ids, :attn_weights.shape[3]].unsqueeze(1)

    attn_weights = F.softmax(attn_weights, dim=3, dtype=torch.float32).to(dtype)

    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, hidden_size)

    attn_output = attn_output @ o_proj.T

    return attn_output

if __name__ == "__main__":
    main()
