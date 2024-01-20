import sys
sys.path.append(".")
from tokenizer import ChatTokenizer
from tests.test_data import test_texts
from util import download
from transformers import AutoTokenizer

def test_tokenizer():
    filename = "data/TinyLlama-1.1B-Chat-v1.0/tokenizer.model"
    url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.model?download=true"
    download(url, filename)

    config_json_url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer_config.json?download=true"
    config_json_filename = "data/TinyLlama-1.1B-Chat-v1.0/tokenizer_config.json"
    download(config_json_url, config_json_filename)

    tokenizer_json_url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json?download=true"
    tokenizer_json_filename = "data/TinyLlama-1.1B-Chat-v1.0/tokenizer.json"
    download(tokenizer_json_url, tokenizer_json_filename)

    prompt = "What is the airspeed velocity of an unladen swallow?"
    prompt = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"

    my_tokenizer = ChatTokenizer(filename)

    auto_tokenizer = AutoTokenizer.from_pretrained("data/TinyLlama-1.1B-Chat-v1.0")

    assert my_tokenizer.encode(prompt) == auto_tokenizer.encode(prompt)[1:]

    for text in test_texts:
        assert my_tokenizer.encode(text) == auto_tokenizer.encode(text)[1:]

    for text in test_texts:
        my_out = my_tokenizer.decode(my_tokenizer.encode(text))
        auto_out = auto_tokenizer.decode(auto_tokenizer.encode(text))
        # Skipping "<s> " at the beginning of the decoded string
        prefix = "<s> "
        assert auto_out == prefix.rstrip() or auto_out.startswith(prefix), f"auto_out should start with {repr(prefix)} but is {repr(auto_out)}"
        auto_out = auto_out[4:]
        assert my_out == auto_out, f"Decoding failed: {repr(my_out)} vs {repr(auto_out)}"

if __name__ == "__main__":
    test_tokenizer()
