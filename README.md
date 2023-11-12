# SimpleTinyLlama

The goal of this project is to serve as a simpler implementation of [TinyLlama](https://github.com/jzhang38/TinyLlama). The only required dependency is PyTorch.

# Installation and usage

1. Install [PyTorch](https://pytorch.org/).
2. Download and extract this repository.
3. Run `main_chat.py` to chat with the model.
4. Press CTRL + C to interrupt the response.
5. Press CTRL + C again to exit the program.

# Notes

- Only inference is supported. Training is not supported.
- Chat history is currently not supported.
- Key/value caching is currently not supported.
- For simplicity and compatibility with older hardware, the default data type is `torch.float32`. It is trivial to convert the `state_dict` weights to `torch.float16` or `torch.bfloat16` if available. The output seems to be identical.
- This project includes a pure Python implementation of a subset of the Sentencepiece tokenizer. It is not as fast as the C++ implementation, but it is sufficient for this project.