import sys
sys.path.append(".")
from tokenizer import ChatTokenizer
from main import llama
import util
import torch

def test_llama_chat():
    prompt = "What is the airspeed velocity of an unladen swallow?"

    model_filename = "data/TinyLlama-1.1B-Chat-v1.0/model.safetensors"
    model_url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors?download=true"
    tokenizer_filename = "data/TinyLlama-1.1B-Chat-v1.0/tokenizer.model"
    tokenizer_url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.model?download=true"

    # Download tokenizer and model
    util.download(tokenizer_url, tokenizer_filename)
    util.download(model_url, model_filename)

    # Choose computation device and data type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the tokenizer
    tokenizer = ChatTokenizer(tokenizer_filename)

    # Load the model weights
    state_dict = util.load_safetensors(model_filename, device)

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
        decoded = tokenizer.decode(token_ids)

        print(repr(decoded))

    # Check if answer is correct
    expected_answer = '<|user|>\nWhat is the airspeed velocity of an unladen swallow?</s>\n<|assistant|>\nThe airspeed velocity of an unladen swallow is approximately 100 km/h (62 mph) at sea level.'
    assert decoded == expected_answer

if __name__ == "__main__":
    test_llama_chat()
