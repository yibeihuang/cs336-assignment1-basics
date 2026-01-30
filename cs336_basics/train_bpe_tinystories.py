import argparse
import json

import cs336_basics.bpe_tokenizer as bpe_tokenizer

def bytes_to_unicode():
    """
    Returns a mapping of all 256 bytes to printable unicode characters.
    Derived from OpenAI's GPT-2 implementation.
    """
    # Standard printable ranges to avoid control characters
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="Path to the training corpus text file")
    parser.add_argument("--vocab_size", type=int, default=10000,
        help="Target vocabulary size")
    parser.add_argument("--output_vocab_path",
        default="data/tinystories_vocab.json",
        help="Path to the output vocabulary file")
    parser.add_argument("--output_merges_path",
        default="data/tinystories_merges.txt",
        help="Path to the output merges file (one merge per line: token1 token2)")
    args = parser.parse_args()

    vocab, merges = bpe_tokenizer.train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
    )
    # Vocab is dict[int, bytes]; dump as dict[str, str] for JSON (token id -> decoded string; invalid bytes → replacement char)
    # vocab_serializable = {str(k): v.decode('utf-8') for k, v in vocab.items()}
    byte_encoder = bytes_to_unicode()
    vocab_serializable = { token_id : "".join(byte_encoder[b] for b in token_bytes)
               for token_id, token_bytes in vocab.items() }
    with open(args.output_vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, indent=0)
    # Merges: one merge per line, space-separated tokens (UTF-8 decoded)
    with open(args.output_merges_path, "w", encoding="utf-8") as f:
        for left, right in merges:
            left_str = "".join(byte_encoder[b] for b in left)
            right_str = "".join(byte_encoder[b] for b in right)
            line = f"{left_str} {right_str}\n"
            f.write(line)
