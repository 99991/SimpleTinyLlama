from typing import List, Dict
from heapq import heappush, heappop
import struct
import re

class TokenNode:
    # Linked list node for token merging
    def __init__(self, value: float, index: int) -> None:
        self.value: float = value
        self.index: int = index
        self.prev: TokenNode = None
        self.next: TokenNode = None

class Tokenizer:
    def __init__(self, filename: str) -> None:
        self.tokens: List[str] = []
        self.byte_tokens: List[bytes] = []
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
                byte_token = f.read(token_size)
                token = byte_token.decode("utf-8")
                assert f.read(1) == b"\x15"
                score = struct.unpack("<f", f.read(4))[0]
                if f.tell() != end:
                    assert f.read(1) == b"\x18"
                    token_type = f.read(1)
                    assert token_type in b"\x02\x03\x06"
                assert f.tell() == end

                if re.match("^<0x[0-9A-F][0-9A-F]>$", token):
                    byte = int(token[1:-1], 16)
                    byte_token = bytes([byte])

                token_id = len(self.tokens)
                self.tokens.append(token)
                self.byte_tokens.append(byte_token)
                self.scores.append(score)
                self.token_ids[token] = token_id
                self.token_scores[token] = score

    def encode_as_tokens(self, text: str) -> List[str]:
        if len(text) == 0: return []

        # Prepand and replace whitespace with unicode underscore character
        text = "▁" + text.replace(" ", "▁")

        return self.encode_as_tokens_raw(text)

    def text_as_unigram_tokens(self, text: str) -> List[str]:
        # Split a given text into the smallest possible tokens
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
        # First, split the text into the smallest possible tokens.
        # Next, merge the two tokens with the highest score.
        # That token pair is found using a heap (priority queue).
        # Repeat until no more tokens can be merged.
        queue = []

        # Split text into smallest possible tokens
        unigram_tokens = self.text_as_unigram_tokens(text)

        # Create linked list of tokens and their index (not linked yet)
        token_nodes = [TokenNode(c, i) for i, c in enumerate(unigram_tokens)]

        def link_and_enqueue(a, b):
            # Link and enqueue two token nodes
            a.next = b
            b.prev = a

            merged = a.value + b.value

            # Check if merged token exists
            if merged in self.token_scores:
                score = self.token_scores[merged]

                # Enqueue merged token, sorted by negative score and
                # then by index as a tie breaker in case of same score.
                heappush(queue, (-score, a.index, merged, a, b))

        # Link neighboring token nodes
        for a, b in zip(token_nodes, token_nodes[1:]):
            link_and_enqueue(a, b)

        first = token_nodes[0]

        # While there are still tokens to merge
        while queue:
            score, _, token, a, b = heappop(queue)

            # Check if tokens have not been merged yet
            if a.value and b.value:
                # Mark tokens as merged
                a.value = None
                b.value = None

                # Create merged token node
                node = TokenNode(token, a.index)

                # Link merged token to previous token if it exists
                if a.prev:
                    link_and_enqueue(a.prev, node)
                else:
                    # Update first node of our linked list if it was merged
                    first = node

                # Link merged token to next token if it exists
                if b.next:
                    link_and_enqueue(node, b.next)

        # Collect linked tokens
        tokens = []
        while first:
            tokens.append(first.value)
            first = first.next

        return tokens

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

def replace_hex(piece):
    # Replace hex strings like "<0x41>"" with the corresponding character
    if re.match("^<0x[0-9A-F][0-9A-F]>$", piece):
        piece = chr(int(piece[1:-1], 16))
    return piece

class ChatTokenizer(Tokenizer):
    def __init__(self, filename: str) -> None:
        super().__init__(filename)

        # Beginning of sentence token
        self.bos = "<s>"
        self.bos_id = self.token_ids[self.bos]

        # Special token ids for chat prompts
        self.pad_token_id = 32000
        self.start_token_id = 32001
        self.end_token_id = 32002

        self.special_token_ids = {
            "<|im_start|>": self.start_token_id,
            "<|im_end|>": self.end_token_id,
        }

        self.special_tokens = {v:k for k, v in self.special_token_ids.items()}


    def encode(self, text: str) -> List[int]:
        # Encode a string into a list of token ids

        # Handle special strings (<|im_start|>, <|im_end|>, etc.) separately
        pattern = "|".join(re.escape(token) for token in self.special_token_ids)
        parts = re.split("(" + pattern + ")", text)

        parts = [part for part in parts if part]

        # Prepend beginning of sentence token
        token_ids = [self.bos_id]

        # Encode each part as a list of token ids
        for part in parts:
            if part in self.special_token_ids:
                # Special tokens are encoded separately
                token_ids.append(self.special_token_ids[part])
            else:
                # Encode text using the tokenizer
                for token in self.encode_as_tokens(part):
                    token_ids.append(self.token_ids[token])

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        # Decode a list of token ids into a string
        assert token_ids[:1] == [self.bos_id]
        token_ids = token_ids[1:]

        tokens = []
        i = 0
        n = len(token_ids)
        while i < n:
            # Decode special tokens
            while i < n and token_ids[i] in self.special_tokens:
                tokens.append(self.special_tokens[token_ids[i]].encode("utf-8"))
                i += 1

            # Decode regular text
            tmp = []
            while i < n and token_ids[i] not in self.special_tokens:
                tmp.append(self.byte_tokens[token_ids[i]])
                i += 1

            # Remove leading underscore
            if tmp:
                magic_underscore = "▁".encode("utf-8")
                assert tmp[0].startswith(magic_underscore)
                tmp[0] = tmp[0][len(magic_underscore):]

            tokens.extend(tmp)

        text = b"".join(tokens).decode("utf-8")

        # Replace underscores with spaces
        text = text.replace("▁", " ")

        return text
