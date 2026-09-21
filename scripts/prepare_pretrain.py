"""Corpus text -> a vocabulary and a packed array of token ids.

Runs once and caches, because it is the slow half: tokenizer training and encoding cost far more
than a training step, and the training script should not pay them on every restart.

The corpus directory is passed in rather than imported from dataforge, so dataforge stays deletable.

    python3 scripts/prepare_pretrain.py --corpus data/corpus --out artifacts
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selfagent.config import ModelConfig  # noqa: E402
from selfagent.tokenizer.vocab import CLS_ID, SEP_ID  # noqa: E402
from selfagent.tokenizer.wordpiece import WordPiece  # noqa: E402

# Held out to report validation loss. Drawn at random rather than taken from the tail, because the
# corpus is written one source at a time and the tail is a single source.
VALIDATION_FRACTION = 0.01
VALIDATION_SEED = 0

_REPORT_EVERY_MEGABYTES = 50
_ENCODE_BUFFER_TOKENS = 5_000_000


def corpus_files(corpus_dir):
    files = sorted(Path(corpus_dir).glob("*.txt"))
    if not files:
        raise SystemExit(f"no .txt files in {corpus_dir}; run dataforge with --process first")
    return files


def lines(files, log):
    """Every line of every corpus file, reporting progress by volume read."""
    read = 0
    next_report = _REPORT_EVERY_MEGABYTES
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                read += len(line)
                if read // (1024 * 1024) >= next_report:
                    log(f"    {read // (1024 * 1024)} MB read")
                    next_report += _REPORT_EVERY_MEGABYTES
                yield line


def pack(token_ids, sequence_length):
    """Ids cut into sequences of exactly `sequence_length`, each wrapped in [CLS] and [SEP].

    Sequences run across document boundaries, as BERT's do: the alternative wastes most of a
    128-token window on the many short documents this corpus contains.
    """
    body = sequence_length - 2
    usable = len(token_ids) // body * body
    rows = token_ids[:usable].reshape(-1, body)
    starts = np.full((len(rows), 1), CLS_ID, dtype=rows.dtype)
    ends = np.full((len(rows), 1), SEP_ID, dtype=rows.dtype)
    return np.hstack([starts, rows, ends])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--vocab-size", type=int, default=ModelConfig.vocab_size)
    parser.add_argument("--sequence-length", type=int, default=ModelConfig.max_text_length)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = corpus_files(args.corpus)
    megabytes = sum(path.stat().st_size for path in files) / (1024 * 1024)
    print(f"{len(files)} corpus files, {megabytes:,.1f} MB")

    vocabulary_path = out / "tokenizer.json"
    if vocabulary_path.exists() and WordPiece.matches_normalization(vocabulary_path):
        tokenizer = WordPiece.load(vocabulary_path)
        print(f"vocabulary: reusing {vocabulary_path} ({tokenizer.vocab_size:,} pieces)")
    else:
        if vocabulary_path.exists():
            print(f"vocabulary: {vocabulary_path} predates the current normalisation, relearning")
        print(f"vocabulary: training {args.vocab_size:,} pieces")
        started = time.monotonic()
        tokenizer = WordPiece.train(lines(files, print), args.vocab_size, log=print)
        tokenizer.save(vocabulary_path)
        print(f"  {tokenizer.vocab_size:,} pieces in {time.monotonic() - started:.0f}s")

    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise SystemExit(f"vocab of {tokenizer.vocab_size} does not fit the uint16 id array")

    print("encoding")
    started = time.monotonic()
    # Buffered into few large arrays rather than one per line: the corpus has millions of lines,
    # and an array each would cost more in object overhead than the ids themselves.
    chunks, buffer, total = [], [], 0
    for line in lines(files, print):
        buffer.extend(tokenizer.encode(line))
        if len(buffer) >= _ENCODE_BUFFER_TOKENS:
            chunks.append(np.asarray(buffer, dtype=np.uint16))
            total += len(buffer)
            buffer.clear()
    if buffer:
        chunks.append(np.asarray(buffer, dtype=np.uint16))
        total += len(buffer)
    print(f"  {total:,} tokens in {time.monotonic() - started:.0f}s")

    sequences = pack(np.concatenate(chunks), args.sequence_length)
    del chunks
    print(f"  {len(sequences):,} sequences of {args.sequence_length}")

    generator = np.random.default_rng(VALIDATION_SEED)
    order = generator.permutation(len(sequences))
    held_out = max(1, int(len(sequences) * VALIDATION_FRACTION))
    np.save(out / "pretrain_validation.npy", sequences[order[:held_out]])
    np.save(out / "pretrain_train.npy", sequences[order[held_out:]])
    print(f"  {len(sequences) - held_out:,} train, {held_out:,} validation -> {out}/")


if __name__ == "__main__":
    main()
