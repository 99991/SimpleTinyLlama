import sys
sys.path.append(".")
from tokenizer import ChatTokenizer
from tests.test_data import test_texts
from util import download
from transformers import AutoTokenizer

def test_tokenizer():
    url = "https://huggingface.co/PY007/TinyLlama-1.1B-Chat-v0.3/resolve/main/tokenizer.model?download=true"
    filename = "../data/TinyLlama-1.1B-Chat-v0.3/tokenizer.model"
    download(url, filename)

    config_url = "https://huggingface.co/PY007/TinyLlama-1.1B-Chat-v0.3/resolve/main/tokenizer_config.json?download=true"
    config_filename = "../data/TinyLlama-1.1B-Chat-v0.3/tokenizer_config.json"
    download(config_url, config_filename)

    tokenizer_json_url = "https://huggingface.co/PY007/TinyLlama-1.1B-Chat-v0.3/resolve/main/tokenizer.json?download=true"
    tokenizer_json_filename = "../data/TinyLlama-1.1B-Chat-v0.3/tokenizer.json"
    download(tokenizer_json_url, tokenizer_json_filename)

    prompt = "What is the airspeed velocity of an unladen swallow?"
    prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    my_tokenizer = ChatTokenizer(filename)

    auto_tokenizer = AutoTokenizer.from_pretrained("data/TinyLlama-1.1B-Chat-v0.3")

    assert my_tokenizer.encode(prompt) == auto_tokenizer.encode(prompt)

    for text in test_texts:
        assert my_tokenizer.encode(text) == auto_tokenizer.encode(text)

    for text in test_texts:
        my_out = my_tokenizer.decode(my_tokenizer.encode(text))
        auto_out = auto_tokenizer.decode(auto_tokenizer.encode(text))
        auto_out = auto_out[4:]  # skipping "<s> " at the beginning of the decoded string
        if not my_out == auto_out:
            # print(f"{my_out} vs {auto_out}")
            assert False

if __name__ == "__main__":
    test_tokenizer()
