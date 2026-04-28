import re
import json
import os
from typing import List, Dict, Tuple
import numpy as np
import csv


def clean_text(text: str) -> str:
    text = text.lower()
    # remove non-alphanumeric characters (keep spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return text.split()


def build_vocab(tokenized_texts: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for tokens in tokenized_texts:
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

    # start indices: 0 -> <PAD>, 1 -> <UNK>
    vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        if c < min_freq:
            continue
        vocab[word] = idx
        idx += 1
    return vocab


def tokens_to_sequence(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    unk = vocab.get("<UNK>", 1)
    return [vocab.get(t, unk) for t in tokens]


def pad_sequences(sequences: List[List[int]], max_len: int, pad_idx: int = 0) -> np.ndarray:
    arr = np.full((len(sequences), max_len), pad_idx, dtype=np.int32)
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        if len(seq) <= max_len:
            arr[i, : len(seq)] = seq
        else:
            arr[i, :] = seq[:max_len]
    return arr


def process_dataset(csv_path: str,
                    out_dir: str = "processed",
                    max_len: int = 128,
                    train_frac: float = 0.7,
                    val_frac: float = 0.15,
                    min_freq: int = 1) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    reviews = []
    ratings = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            reviews.append(r["review"])
            ratings.append(int(r["rating"]))

    # shuffle
    rng = np.random.default_rng(42)
    idxs = np.arange(len(reviews))
    rng.shuffle(idxs)
    reviews = [reviews[i] for i in idxs]
    ratings = [ratings[i] for i in idxs]

    n = len(reviews)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    train_reviews = reviews[:n_train]
    val_reviews = reviews[n_train : n_train + n_val]
    test_reviews = reviews[n_train + n_val :]

    train_ratings = ratings[:n_train]
    val_ratings = ratings[n_train : n_train + n_val]
    test_ratings = ratings[n_train + n_val :]

    # tokenize and clean
    train_tokens = [tokenize(clean_text(t)) for t in train_reviews]
    val_tokens = [tokenize(clean_text(t)) for t in val_reviews]
    test_tokens = [tokenize(clean_text(t)) for t in test_reviews]

    # build vocab from train only
    vocab = build_vocab(train_tokens, min_freq=min_freq)

    # convert to sequences
    train_seq = [tokens_to_sequence(toks, vocab) for toks in train_tokens]
    val_seq = [tokens_to_sequence(toks, vocab) for toks in val_tokens]
    test_seq = [tokens_to_sequence(toks, vocab) for toks in test_tokens]

    # pad / truncate
    X_train = pad_sequences(train_seq, max_len)
    X_val = pad_sequences(val_seq, max_len)
    X_test = pad_sequences(test_seq, max_len)

    y_train = np.array(train_ratings, dtype=np.int32)
    y_val = np.array(val_ratings, dtype=np.int32)
    y_test = np.array(test_ratings, dtype=np.int32)

    # save
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "y_val.npy"), y_val)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)

    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as vf:
        json.dump(vocab, vf, ensure_ascii=False)

    meta = {
        "n_total": n,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "max_len": max_len,
        "vocab_size": len(vocab),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False)

    return {"out_dir": out_dir, "meta": meta}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", default="processed")
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()
    info = process_dataset(args.csv, out_dir=args.out_dir, max_len=args.max_len)
    print(info)
