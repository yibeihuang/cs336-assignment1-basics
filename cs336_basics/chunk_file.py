from typing import Iterator


def iter_chunks_by_string(
    input_path: str, split_token: str, read_size: int
) -> Iterator[str]:
    """
    Stream the file and yield string chunks split by split_token.
    Opens in text mode so the file handle handles UTF-8 decoding; we never
    split in the middle of a character.
    """
    text_buf = ""

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            block = f.read(read_size)
            if not block:
                break
            text_buf += block

            if not split_token:
                continue

            # Emit all complete chunks (text before each occurrence of split_token)
            while split_token in text_buf:
                idx = text_buf.index(split_token)
                chunk = text_buf[:idx]
                text_buf = text_buf[idx + len(split_token) :]
                if chunk:
                    yield chunk

        # Last chunk (after final token or whole buffer if no token)
        if text_buf:
            yield text_buf
