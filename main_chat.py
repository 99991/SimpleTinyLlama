from tokenizer import ChatTokenizer
from llama import llama
from util import download
import torch

def main():
    # Where to download the model and tokenizer and where to store it
    model_url = "https://huggingface.co/PY007/TinyLlama-1.1B-Chat-v0.3/resolve/main/pytorch_model.bin?download=true"
    model_filename = "data/TinyLlama-1.1B-Chat-v0.3/pytorch_model.bin"
    tokenizer_url = "https://huggingface.co/PY007/TinyLlama-1.1B-Chat-v0.3/resolve/main/tokenizer.model?download=true"
    tokenizer_filename = "data/TinyLlama-1.1B-Chat-v0.3/tokenizer.model"

    # Download tokenizer and model
    download(tokenizer_url, tokenizer_filename)
    download(model_url, model_filename)

    # Choose computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model weights
    state_dict = torch.load(model_filename, map_location=device)

    # Question loop
    while True:
        # Ask for a question
        try:
            prompt = input("Ask a question:\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        try:
            # Convert to chat prompt format
            prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

            # Convert prompt to a list of token ids
            tokenizer = ChatTokenizer(tokenizer_filename)
            token_ids = tokenizer.encode(prompt)

            # Generate tokens one by one
            while len(token_ids) < 2048:
                # Upload token ids to device
                token_ids_torch = torch.tensor([token_ids], device=device)

                # Predict logits
                logits = llama(token_ids_torch, state_dict)

                # Choose the most likely token
                new_token_id = logits[0, -1].argmax().item()

                # Stop if we reach the special end token, which indicates the end of the response
                if new_token_id == tokenizer.end_token_id:
                    break

                # Append the new token to the list of tokens
                token_ids.append(new_token_id)

                # Decode all tokens so far
                decoded = tokenizer.decode(token_ids)

                response_start = "<|im_start|>assistant\n"

                # Print response after <|im_start|>assistant\n
                if response_start in decoded:
                    response = decoded.split(response_start)[1]
                
                    print(response)
        except KeyboardInterrupt:
            print("Response interrupted")

if __name__ == "__main__":
    main()
