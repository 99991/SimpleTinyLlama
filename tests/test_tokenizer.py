import sys
sys.path.append("..")
from tokenizer import Tokenizer
from sentencepiece import SentencePieceProcessor

def test_tokenizer():
    filename = "../data/TinyLlama-1.1B-Chat-v0.3/tokenizer.model"

    sp_tokenizer = SentencePieceProcessor(filename)
    my_tokenizer = Tokenizer(filename)

    assert sp_tokenizer.vocab_size() == len(my_tokenizer.tokens)

    for token_id in range(len(sp_tokenizer)):
        assert sp_tokenizer.id_to_piece(token_id) == my_tokenizer.tokens[token_id]
        assert sp_tokenizer.get_score(token_id) == my_tokenizer.scores[token_id]

    test_texts = [
        "",
        "foo",
        "quack",
        "Hello, World!",
        "The quick brown fox jumps over the lazy dog.",
        "SPHINX OF BLACK QUARTZ, JUDGE MY VOW!",
        "It is very rare to see hippos hiding in trees because they are very good at it.",
        "☃",
        "🤗",
        r"¯\_(ツ)_/¯",
        "⸜(｡˃ ᵕ ˂ )⸝♡"
        "𓆝 𓆟 𓆞 𓆝 𓆟",
        "H₂ + O₂ ⇌ 2H₂O",
        "读万卷书不如行万里路",
        "猿も木から落ちる",
        """
#include <stdio.h>

int main(){
    printf("Hello, World!");
    return EXIT_SUCCESS;
}
        """,
        "Viele Grüße aus Köln und Düsseldorf :)",
    ]

    for text in test_texts:
        assert sp_tokenizer.EncodeAsPieces(text) == my_tokenizer.encode_as_tokens(text)

if __name__ == "__main__":
    test_tokenizer()
