import os
from itertools import chain
from multiprocessing import Pool
from timeit import repeat
from typing import BinaryIO

import regex as re


# Helper Function
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> tuple[list[int], int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    token_len = len(split_special_token)

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return (sorted(set(chunk_boundaries)), file_size)


# Compile the regex for speed
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
pattern = re.compile(PAT)


# Tokenization (BPE)
class BPETokenizer:
    # Special tokens contain chunk_boundary
    def __init__(self, filepath: str, special_tokens: list[str], chunk_boundary: bytes = b"<|endoftext|>"):
        self.special_tokens = [t.encode("utf-8") for t in special_tokens]
        self.filepath = filepath
        self.chunk_boundary = chunk_boundary
        self.pre_split_pat = re.compile("|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)))
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []
        self.pre_vocab: list[bytes] = []

    # Per Process Pretokenization
    def _pretokenize_batch(self, args: tuple[str, tuple[int, int]]) -> list[bytes]:
        # Debug Info
        # print(os.getpid())
        filename, boundary = args
        start, end = boundary

        # Give each process an independent fd, so that the cursor does not mess up with one another
        with open(filename, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

        special_strs = [t.decode("utf-8") for t in self.special_tokens]

        if not special_strs:
            return [m.encode("utf-8") for m in pattern.findall(chunk)]

        special_strs.sort(key=len, reverse=True)

        special_pattern = "(" + "|".join(re.escape(t) for t in special_strs) + ")"
        special_pattern = re.compile(special_pattern)

        results: list[bytes] = []

        pos = 0

        for match in special_pattern.finditer(chunk):
            s, e = match.span()

            t = chunk[pos:s]

            if t:
                results.extend(token.encode("utf-8") for token in pattern.findall(t))

            results.append(match.group().encode("utf-8"))

            pos = e

        tail = chunk[pos:]

        if tail:
            results.extend(token.encode("utf-8") for token in pattern.findall(tail))

        return results

    # Pretokenization
    def pretokenize(self):
        with open(self.filepath, "rb") as f:
            num_processes = 8
            boundaries, filesize = find_chunk_boundaries(f, num_processes, self.chunk_boundary)

        tasks = [(self.filepath, (s, e)) for s, e in zip(boundaries[:-1], boundaries[1:])]

        # p.map gets list[list[T]]
        with Pool(num_processes) as p:
            results = p.map(self._pretokenize_batch, tasks)

        # chain.from_iterable gets iterable of iterables chained, then iterated
        self.pre_vocab = list(chain.from_iterable(results))

    def train(self, vocab_size: int):
        pass


if __name__ == "__main__":
    tokenizer = BPETokenizer(
        "/Users/aeilot/Developer/learning/CS336/assignment1-basics/data/owt_valid.txt",
        ["<|endoftext|>"],
    )
    tokenizer.pretokenize()

    test_out_path = "/Users/aeilot/Developer/learning/CS336/assignment1-basics/data/test_pretokenizer.txt"

    with open(test_out_path, "w", encoding="utf-8") as f:
        sample_tokens = tokenizer.pre_vocab

        output = b"|".join(sample_tokens)
        f.write(output.decode("utf-8", errors="replace"))

        print(f"Write Success. Total pre-tokens generated: {len(tokenizer.pre_vocab)}")
