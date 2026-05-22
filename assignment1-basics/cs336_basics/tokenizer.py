import os
import pickle
import token
from collections import Counter, defaultdict
from itertools import chain, islice
from multiprocessing import Pool
from pathlib import Path
from timeit import repeat
from tokenize import tok_name
from typing import BinaryIO, Iterable, Iterator

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
# GPT-2 Styles
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
pattern = re.compile(PAT)


# Tokenization (BPE)
class Tokenizer:
    # Special tokens contain chunk_boundary
    def __init__(self, special_tokens: list[str], chunk_boundary: bytes = b"<|endoftext|>"):
        self.special_tokens = [t.encode("utf-8") for t in special_tokens]
        self.chunk_boundary = chunk_boundary
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        self.vocab.extend(self.special_tokens)
        self.idx: dict[bytes, int] = {v: i for i, v in enumerate(self.vocab)}
        self.merges: list[tuple[bytes, bytes]] = []
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {}
        self._pretokens: dict[tuple[int, ...], int] = {}
        self._pair_index: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)

    # Per Process Pretokenization
    def _pretokenize_batch(self, args: tuple[str, tuple[int, int]]) -> Counter[bytes]:
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
            return Counter([m.encode("utf-8") for m in pattern.findall(chunk)])

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
                # results.extend(token.encode("utf-8") for token in t.split())

            pos = e

        tail = chunk[pos:]

        if tail:
            results.extend(token.encode("utf-8") for token in pattern.findall(tail))
            # results.extend(token.encode("utf-8") for token in tail.split())

        return Counter(results)

    # Pretokenization
    def _pretokenize(self, filepath: str, num_processes: int = 8):
        with open(filepath, "rb") as f:
            boundaries, filesize = find_chunk_boundaries(f, num_processes, self.chunk_boundary)

        tasks = [(filepath, (s, e)) for s, e in zip(boundaries[:-1], boundaries[1:])]

        pre_token_count: Counter[bytes] = Counter()

        # p.map gets list[list[T]]
        with Pool(num_processes) as p:
            counters = p.map(self._pretokenize_batch, tasks)

            for c in counters:
                pre_token_count.update(c)

        # Make Words
        self._pretokens.clear()

        # Words are tuple[int, ...] representing token sequence; special tokens are not words
        for token_bytes, freq in pre_token_count.items():
            token_ids = tuple(token_bytes)
            self._pretokens[token_ids] = freq

    # Count Words
    def _count_pair(self) -> Counter[tuple[int, int]]:
        counts: Counter[tuple[int, int]] = Counter()
        self._pair_index.clear()

        for word, freq in self._pretokens.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                self._pair_index[pair].add(word)
                counts[pair] += freq

        return counts

    def train(self, filepath: str, vocab_size: int, num_processes: int = 8):
        # Pretokenize
        self._pretokenize(filepath, num_processes)

        # Chunk pretoken counter
        pair_counts: Counter[tuple[int, int]] = self._count_pair()

        while len(self.vocab) < vocab_size:
            if not len(pair_counts):
                break

            # Get most frequent pair
            most_freq, best_freq = max(
                pair_counts.items(),
                key=lambda x: (x[1], self.vocab[x[0][0]], self.vocab[x[0][1]]),
            )
            new_id = len(self.vocab)
            new_token = self.vocab[most_freq[0]] + self.vocab[most_freq[1]]

            if best_freq <= 0:
                break

            self.merges.append((self.vocab[most_freq[0]], self.vocab[most_freq[1]]))
            self.vocab.append(new_token)
            self.idx[new_token] = new_id

            affected_words = list(self._pair_index.get(most_freq, ()))

            for word in affected_words:
                freq = self._pretokens.pop(word, 0)

                if freq == 0:
                    continue

                # Remove old word stats
                old_seen: set[tuple[int, int]] = set()

                for j in range(len(word) - 1):
                    old_pair = (word[j], word[j + 1])
                    pair_counts[old_pair] -= freq
                    old_seen.add(old_pair)

                    if pair_counts[old_pair] <= 0:
                        del pair_counts[old_pair]

                for old_pair in old_seen:
                    words = self._pair_index.get(old_pair)

                    if words is not None:
                        words.discard(word)

                        if not words:
                            del self._pair_index[old_pair]

                # Merge Words
                new_word_list = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == most_freq[0] and word[i + 1] == most_freq[1]:
                        new_word_list.append(new_id)
                        i += 2
                    else:
                        new_word_list.append(word[i])
                        i += 1

                new_word = tuple(new_word_list)

                # if new_word already exists, remove its old stats first
                existing_freq = self._pretokens.pop(new_word, 0)

                if existing_freq:
                    existing_seen: set[tuple[int, int]] = set()

                    for j in range(len(new_word) - 1):
                        existing_pair = (new_word[j], new_word[j + 1])
                        pair_counts[existing_pair] -= existing_freq
                        existing_seen.add(existing_pair)

                        if pair_counts[existing_pair] <= 0:
                            del pair_counts[existing_pair]

                    for existing_pair in existing_seen:
                        words = self._pair_index.get(existing_pair)

                        if words is not None:
                            words.discard(new_word)

                            if not words:
                                del self._pair_index[existing_pair]

                total_freq = freq + existing_freq
                self._pretokens[new_word] = total_freq

                # add new word stats
                new_seen: set[tuple[int, int]] = set()

                for j in range(len(new_word) - 1):
                    new_pair = (new_word[j], new_word[j + 1])
                    pair_counts[new_pair] += total_freq
                    new_seen.add(new_pair)

                for new_pair in new_seen:
                    self._pair_index.setdefault(new_pair, set()).add(new_word)
                    import pickle
                    from pathlib import Path

        self.merge_ranks = {merge: i for i, merge in enumerate(self.merges)}

    @classmethod
    def load(cls, vocab: list[bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[bytes]) -> "Tokenizer":
        tokenizer = cls(
            special_tokens=[t.decode("utf-8") for t in special_tokens],
        )

        tokenizer.vocab = vocab
        tokenizer.merges = merges
        tokenizer.special_tokens = special_tokens
        tokenizer.idx = {token: i for i, token in enumerate(tokenizer.vocab)}
        tokenizer.merge_ranks = {merge: i for i, merge in enumerate(tokenizer.merges)}
        return tokenizer

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        if special_tokens is None:
            special_tokens = []

        return cls.load(vocab, merges, [t.encode("utf-8") for t in special_tokens])

    def to_files(self, vocab_filepath, merges_filepath):
        with open(vocab_filepath, "wb") as f:
            pickle.dump(self.vocab, f)

        with open(merges_filepath, "wb") as f:
            pickle.dump(self.merges, f)

    def decode(self, ids: list[int]) -> str:
        tokens = [self.vocab[i] for i in ids]
        return b"".join(tokens).decode("utf-8", errors="ignore")

    def encode(self, text: str) -> list[int]:
        # Special Tokens
        special_strs = [t.decode("utf-8") for t in self.special_tokens]

        if not special_strs:
            raw_chunks = [token.encode("utf-8") for token in pattern.findall(text)]
        else:
            special_strs.sort(key=len, reverse=True)
            special_pattern = re.compile("(" + "|".join(re.escape(t) for t in special_strs) + ")")

            raw_chunks = []
            pos = 0
            for match in special_pattern.finditer(text):
                s, e = match.span()
                t = text[pos:s]
                if t:
                    raw_chunks.extend(token.encode("utf-8") for token in pattern.findall(t))
                raw_chunks.append(match.group().encode("utf-8"))
                pos = e
            tail = text[pos:]
            if tail:
                raw_chunks.extend(token.encode("utf-8") for token in pattern.findall(tail))

        final_ids = []
        for chunk in raw_chunks:
            if chunk in self.special_tokens:
                final_ids.append(self.idx[chunk])
                continue

            # For regular text chunks, break down into initial base bytes
            ids = [self.idx[bytes([b])] for b in chunk]

            while len(ids) >= 2:
                # Find the pair with the lowest merge rank (highest priority)
                best_pair = None
                best_rank = float("inf")

                for i in range(len(ids) - 1):
                    pair = (self.vocab[ids[i]], self.vocab[ids[i + 1]])
                    rank = self.merge_ranks.get(pair, float("inf"))
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = (i, pair)

                if best_pair is None:
                    break  # No more mergeable pairs exist

                # Merge the highest priority pair found
                idx_to_merge, pair_bytes = best_pair
                merged_token_bytes = pair_bytes[0] + pair_bytes[1]
                merged_id = self.idx[merged_token_bytes]

                new_ids = []
                i = 0
                while i < len(ids):
                    if i == idx_to_merge:
                        new_ids.append(merged_id)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                ids = new_ids

            final_ids.extend(ids)

        return final_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)


import time
import tracemalloc

if __name__ == "__main__":
    data_path = "/Users/aeilot/Developer/learning/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"

    # For profiling, Time and Memory tracing
    # start_time = time.time()
    # tracemalloc.start()

    tokenizer = Tokenizer(["<|endoftext|>"], b"<|endoftext|>")
    tokenizer.train(data_path, 10000)

    # current, peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # end_time = time.time()

    # print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    # print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    # print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")

    vocab: dict[int, bytes] = {i: token for i, token in enumerate(tokenizer.vocab)}

    output_path = "/Users/aeilot/Developer/learning/CS336/assignment1-basics/data/test_pretokenizer.txt"

    with open(output_path, "w") as f:
        for k, v in vocab.items():
            f.write(f"{k}, {v}\n")

    longest_token = max(tokenizer.vocab, key=len)

    print(longest_token)
    print(len(longest_token))

    # Save To File
    tokenizer.to_files("vocab.pkl", "merges.pkl")
