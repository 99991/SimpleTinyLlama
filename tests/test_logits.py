import sys
sys.path.append(".")
import torch
import util
from main import llama, ChatTokenizer
from transformers import LlamaForCausalLM

def compute_logits(token_ids):
    model_filename = "data/TinyLlama-1.1B-Chat-v1.0/model.safetensors"
    device = torch.device("cpu")
    # Conver to float32 because that is what LlamaForCausalLM does
    state_dict = util.load_safetensors(model_filename, device, new_dtype=torch.float32)
    # position_ids = [[0, 1, 2]]
    position_ids = torch.arange(token_ids.shape[-1]).view(1, -1)
    cache = {}
    logits = llama(token_ids, position_ids, cache, state_dict)
    return logits

def compute_expected_logits(token_ids):
    model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    logits = model(token_ids).logits
    return logits

def test():
    token_ids = torch.tensor([[1, 1053, 12655]])

    logits = compute_logits(token_ids)
    expected_logits = compute_expected_logits(token_ids)

    mse = (logits - expected_logits).pow(2).mean().item()

    # Note that the MSE is only exactly 0.0 if both the data types and the
    # computation devices are identical.
    assert mse == 0.0, f"Expected mean squared error of 0.0 for logits, but got {mse}"

if __name__ == "__main__":
    test()
