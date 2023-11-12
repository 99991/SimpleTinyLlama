import sys
sys.path.append(".")
from tokenizer import Tokenizer
from util import download
from sentencepiece import SentencePieceProcessor
from tests.test_data import test_texts

def test_tokenizer():
    url = "https://huggingface.co/PY007/TinyLlama-1.1B-Chat-v0.3/resolve/main/tokenizer.model?download=true"
    filename = "data/TinyLlama-1.1B-Chat-v0.3/tokenizer.model"
    download(url, filename)

    sp_tokenizer = SentencePieceProcessor(filename)
    my_tokenizer = Tokenizer(filename)

    assert sp_tokenizer.vocab_size() == len(my_tokenizer.tokens)

    for token_id in range(len(sp_tokenizer)):
        assert sp_tokenizer.id_to_piece(token_id) == my_tokenizer.tokens[token_id]
        assert sp_tokenizer.get_score(token_id) == my_tokenizer.scores[token_id]

    for text in test_texts:
        assert sp_tokenizer.EncodeAsPieces(text) == my_tokenizer.encode_as_tokens(text)

if __name__ == "__main__":
    test_tokenizer()
