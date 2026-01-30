# This is a BPE tokenizer implementation.
import os
import regex as re
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

import psutil
from collections import Counter, defaultdict
from typing import Iterator
from cs336_basics.chunk_file import iter_chunks_by_string


# Special tokens to be added to the vocabulary
NUM_PROCESSES = 8
# When file is larger than this (bytes), use streaming chunking instead of loading all
STREAMING_THRESHOLD_BYTES = 100 * 1024 * 1024  # 100 MiB
STREAM_READ_SIZE = 256 * 1024  # 256 KiB per read


def pre_tokenize(chunk: str, special_tokens: list[str]) -> Iterator[re.Match]:
    """
    Pre-tokenize the training corpus into an iterator of strings.
    """
    special_pattern = "|".join(re.escape(t) for t in special_tokens)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    combined_pat = f"({special_pattern})|{PAT}"
    return re.finditer(combined_pat, chunk)


def process_chunk(special_tokens: list[str], chunk: str):
    tokens = pre_tokenize(chunk, special_tokens)
    return Counter(m.group().encode("utf-8") for m in tokens)


def compute_pair_freqs(splits: dict[bytes, list[bytes]], frequency_dict: dict[bytes, int]):
    """Calculate occurence frequence of each remaining pair.
    """
    pair_freqs = defaultdict(int)
    for word_in_byte, freq in frequency_dict.items():
        split = splits.get(word_in_byte)
        if split is None or len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def merge_pair(a: bytes, b: bytes, splits: dict[bytes, list[bytes]]):
    """Apply merged pair in splits dictionary to update them all.
    Optimized to avoid repeated list slicing.
    """
    merged_token = a + b
    # Only iterate over words that are in splits (excludes special tokens)
    for word_in_byte in splits:
        split = splits[word_in_byte]
        if len(split) <= 1:
            continue

        # Build new split list in-place to avoid repeated slicing
        new_split = []
        i = 0
        while i < len(split):
            if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                new_split.append(merged_token)
                i += 2  # Skip both tokens we just merged
            else:
                new_split.append(split[i])
                i += 1
        splits[word_in_byte] = new_split
    return splits


def train_bpe(input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input corpus.

    Args:
        input_path: Path to a text file containing the training data.
        vocab_size: Final vocabulary size including initial byte vocabulary, vocabulary items produced from mergingand any special tokens.
        special_tokens: List of special tokens to be added to the vocabulary.

    Returns:
        A tuple containing the vocabulary and the merges.
        vocab: dict[int, bytes]: The vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges: list[tuple[bytes, bytes]]: BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
    """
    # Initialize the vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    # Add special tokens starting from index 256
    for idx, special_token in enumerate(special_tokens):
        vocab[256 + idx] = special_token.encode("utf-8")
    
    if len(vocab) >= vocab_size:
        raise ValueError("Maximum final vocabulary size is smaller than or "
        "equal to the initial vocabulary size. Increase the final vocabulary size.")

    # Pre-tokenize the training corpus. Chunk on document boundaries so we never split
    # in the middle of a UTF-8 character. Use the standard document-boundary token.
    split_token = "<|endoftext|>"
    file_size = os.path.getsize(input_path)
    frequency_dict = Counter()
    worker_func = partial(process_chunk, special_tokens)

    if file_size <= STREAMING_THRESHOLD_BYTES:
        # Small/medium file: load all, decode once, split by token, process in parallel
        with open(input_path, "rb") as f:
            raw = f.read()
        full_text = raw.decode("utf-8", errors="ignore")
        chunks = full_text.split(split_token)
        with mp.get_context("spawn").Pool(processes=NUM_PROCESSES) as p:
            for freq_dict in p.imap(worker_func, chunks):
                frequency_dict.update(freq_dict)
    else:
        # Large file: stream with UTF-8-safe decoding, chunk on special token; aggregate chunks then process in parallel
        chunks_list: list[str] = []
        for chunk in iter_chunks_by_string(input_path, split_token, STREAM_READ_SIZE):
            chunks_list.append(chunk)
        worker_func = partial(process_chunk, special_tokens)
        with mp.get_context("spawn").Pool(processes=NUM_PROCESSES) as p:
            for freq_dict in p.imap(worker_func, chunks_list):
                frequency_dict.update(freq_dict)

    special_tokens_bytes = [token.encode('utf-8') for token in special_tokens]
    splits = {
        word: [bytes([c]) for c in word]
        for word in frequency_dict.keys()
        if word not in special_tokens_bytes
    }

    num_merges = vocab_size - len(vocab)
    merges = []
    process = psutil.Process()
    pbar = tqdm(range(num_merges), desc="BPE merges", unit="merge")
    for i in pbar:
        pair_freqs = compute_pair_freqs(splits, frequency_dict)
        if not pair_freqs:
            break

        # Find the pair with maximum frequency
        # Use max() with key function for efficiency - O(n) but cleaner and potentially faster
        # For ties, lexicographically larger pair wins (standard BPE behavior)
        best_pair = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]

        splits = merge_pair(*best_pair, splits)
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        # Show current process RSS in MiB
        rss_mib = process.memory_info().rss / (1024 * 1024)
        pbar.set_postfix(mem_mib=f"{rss_mib:.1f}")

    return (vocab, merges)
