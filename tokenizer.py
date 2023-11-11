from typing import List, Dict
from heapq import heappush, heappop
import struct

class TokenNode:
    def __init__(self, value: float, index: int) -> None:
        self.value: float = value
        self.index: int = index
        self.prev: TokenNode = None
        self.next: TokenNode = None

class Tokenizer:
    def __init__(self, filename: str) -> None:
        self.tokens: List[str] = []
        self.scores: List[float] = []
        self.token_ids: Dict[str, int] = {}
        self.token_scores: Dict[str, float] = {}

        # Read token information from Protobuf file
        with open(filename, "rb") as f:
            while f.read(1) == b"\n":
                size = f.read(1)[0]
                end = f.tell() + size
                assert f.read(1) == b"\n"
                token_size = f.read(1)[0]
                token = f.read(token_size).decode("utf-8")
                assert f.read(1) == b"\x15"
                score = struct.unpack("<f", f.read(4))[0]
                if f.tell() != end:
                    assert f.read(1) == b"\x18"
                    token_type = f.read(1)
                    assert token_type in b"\x02\x03\x06"
                assert f.tell() == end

                token_id = len(self.tokens)
                self.tokens.append(token)
                self.scores.append(score)
                self.token_ids[token] = token_id
                self.token_scores[token] = score

    def encode_as_tokens(self, text: str) -> List[str]:
        if len(text) == 0: return []

        # Prepand and replace whitespace with unicode underscore character
        text = "▁" + text.replace(" ", "▁")

        return self.encode_as_tokens_raw(text)

    def text_as_unigram_tokens(self, text: str) -> List[str]:
        tokens = []
        for i, c in enumerate(text):
            if c in self.token_scores:
                tokens.append(c)
            else:
                # Split unknown characters into bytes
                for x in c.encode("utf-8"):
                    tokens.append(f"<0x{x:02X}>")
        return tokens

    def encode_as_tokens_raw(self, text: str) -> List[str]:
        # TODO comments
        queue = []

        unigram_tokens = self.text_as_unigram_tokens(text)

        pieces = [TokenNode(c, i) for i, c in enumerate(unigram_tokens)]

        def enqueue(a, b):
            a.next = b
            b.prev = a

            piece = a.value + b.value

            if piece in self.token_scores:
                score = self.token_scores[piece]

                heappush(queue, (-score, a.index, piece, a, b))

        for a, b in zip(pieces, pieces[1:]):
            enqueue(a, b)

        first = pieces[0]

        while queue:
            score, _, piece, a, b = heappop(queue)

            if a.value and b.value:
                a.value = None
                b.value = None

                node = TokenNode(piece, a.index)

                if a.prev:
                    enqueue(a.prev, node)
                else:
                    first = node

                if b.next:
                    enqueue(node, b.next)

        pieces = []
        while first:
            pieces.append(first.value)
            first = first.next

        return pieces

    # Same as above, but O(n^2) (without priority queue)
    """
    def encode_as_tokens_raw(self, text: str) -> List[str]:
        tokens = self.text_as_unigram_tokens(text)

        while True:
            max_score = (-float("inf"), -1, "")

            # Merge all consecutive tokens
            for i in range(len(tokens) - 1):
                merged_token = tokens[i] + tokens[i + 1]

                # Check if merged token exists
                if merged_token in self.token_scores:
                    score = self.token_scores[merged_token]
                    # Update token with maximum score
                    if score > max_score[0]:
                        max_score = (score, i, merged_token)

            score, i, token = max_score

            # Done when no merged token could be found
            if i == -1:
                break

            # Replace two consecutive tokens with merged token
            tokens = tokens[:i] + [token] + tokens[i+2:]

        return tokens
    """
