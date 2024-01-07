# SimpleTinyLlama

[TinyLlama](https://github.com/jzhang38/TinyLlama) is a relatively small large language model with impressive capabilities for its size.
The goal of this project is to serve as a simpler implementation of TinyLlama. The only required dependency is PyTorch.

<div align="center">
  <img src="https://github.com/99991/SimpleTinyLlama/assets/18725165/834efd71-e933-407f-92bb-f19979c34188" width="300"/>
</div>

# Installation and usage

1. Install [PyTorch](https://pytorch.org/).
2. Download and extract this repository.
3. Run `main.py` to chat with the llama.
4. Press CTRL + C to interrupt the response.
5. Press CTRL + C again to exit the program.

# Example
![Example of asking how to steal a duck.](https://github.com/99991/SimpleTinyLlama/assets/18725165/a1e3ec73-294e-433a-88e1-8645f46271fd)

# Notes

- CUDA will be used if available, but requires approximately 3 GB of VRAM. If you do not have that much VRAM, you can set the computation device manually in [`main.py`](https://github.com/99991/SimpleTinyLlama/blob/main/main.py#L18).
- Only inference is supported. Training is not supported.
- Chat history is currently not supported.
- This project includes a pure Python implementation of a subset of the Sentencepiece tokenizer. It is not as fast as the C++ implementation, but it is sufficient for this project.

# TODO

- Update tests to work with version 1.0 of TinyLlama (chat format changed).
