"""Tokenizer class
Given a vocabulary and a list of merges, encodes text into integer IDs and decodes integer IDs back into text.
Steps of encoding:
1. Pre-tokenize the text into a list of tokens.
2. Apply merges in the same order as the creation of the merges file.
3. Return the list of token IDs.
Steps of decoding:
1. Convert the list of token IDs into a list of bytes.
2. Convert the list of bytes into a string.
3. Return the string.
"""
import json
import re
import os
from collections.abc import Iterable

from cs336_basics.bpe_tokenizer import pre_tokenize
from cs336_basics.train_bpe_tinystories import bytes_to_unicode


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None) -> None:
        """Initialize a Tokenizer."""
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.byte_encoder = bytes_to_unicode()
        self.vocab_inverse_lookup = {v: k for k, v in self.vocab.items()}
        # Cache for O(1) membership and single-pass split
        self._special_tokens_set = frozenset(special_tokens) if special_tokens else None
        if special_tokens:
            # Longest first so "<|eot|><|eot|>" matches before "<|eot|>"
            sorted_special = sorted(special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(s) for s in sorted_special)
            self._special_token_re = re.compile(f"({pattern})")
        else:
            self._special_token_re = None

    @classmethod
    def from_files(cls, vocab_filepath: str | os.PathLike,
                   merges_filepath: str | os.PathLike,
                   special_tokens: list[str] | None = None) -> "Tokenizer":
        """Construct a Tokenizer from a vocabulary and merges file.

        Args:
            vocab_filepath: Path to the vocabulary file.
            merges_filepath: Path to the merges file.
            special_tokens: List of special tokens.

        Returns:
            A Tokenizer.
        """
        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)
        with open(merges_filepath, "r") as f:
            merges = [tuple(line.split()) for line in f]
        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, token_list: list[bytes]) -> list[int]:
        """Apply merges to a token list (list of single-byte or multi-byte tokens), then return token IDs."""
        result = list(token_list)
        for left, right in self.merges:
            new_result = []
            i = 0
            while i < len(result):
                if i < len(result) - 1 and result[i] == left and result[i + 1] == right:
                    new_result.append(left + right)
                    i += 2
                else:
                    new_result.append(result[i])
                    i += 1
            result = new_result
        return [self.vocab_inverse_lookup[token] for token in result]

    def _bytes_to_token_ids(self, text_bytes: bytes) -> list[int]:
        """Split text into single-byte tokens and apply BPE merges."""
        single_byte_tokens = [bytes([b]) for b in text_bytes]
        return self._apply_merges(single_byte_tokens)

    def _split_by_special_tokens(self, text: str) -> list[str]:
        """Split text into segments of regular text and special tokens (single regex pass)."""
        if not text:
            return []
        if self._special_token_re is None:
            return [text]
        parts = []
        last_end = 0
        for match in self._special_token_re.finditer(text):
            if match.start() > last_end:
                parts.append(text[last_end : match.start()])
            parts.append(match.group())
            last_end = match.end()
        if last_end < len(text):
            parts.append(text[last_end:])
        return parts

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of integers."""
        if self.special_tokens is None:
            ids = []
            for token in pre_tokenize(text, None):
                ids.extend(self._bytes_to_token_ids(token.group().encode("utf-8")))
            return ids

        ids = []
        for part in self._split_by_special_tokens(text):
            if part in self._special_tokens_set:
                ids.append(self.vocab_inverse_lookup[part.encode("utf-8")])
            else:
                for token in pre_tokenize(part, None):
                    ids.extend(self._bytes_to_token_ids(token.group().encode("utf-8")))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """Encode an iterable of strings into a generator of token IDs.
        This is required for memory-efficient tokenization of large files that don't fit in memory.
        
        Args:
            iterable: An iterable of strings.

        Returns:
            A generator that lazily yields token IDs.
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """Decode a list of integers into a string."""
        if not ids:
            return ""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
