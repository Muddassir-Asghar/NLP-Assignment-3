import os
import json
import random
from typing import List, Dict, Any, Optional


def extract_text_and_rating(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Try common fields for review text
    text_keys = ["reviewText", "review_body", "review", "text"]
    rating_keys = ["overall", "stars", "rating"]

    text = None
    for k in text_keys:
        if k in obj and obj[k]:
            text = obj[k]
            break

    rating = None
    for k in rating_keys:
        if k in obj and obj[k] is not None:
            try:
                rating = int(float(obj[k]))
            except Exception:
                pass
            break

    if text is None or rating is None:
        return None
    if not (1 <= rating <= 5):
        return None
    return {"review": text, "rating": rating}


def sample_category_file(file_path: str, requested: int) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    # Support both JSON-lines and array JSON formats
    with open(file_path, "r", encoding="utf-8") as f:
        first = f.read(2)
        f.seek(0)
        if first.strip().startswith("["):
            # JSON array
            data = json.load(f)
            for obj in data:
                item = extract_text_and_rating(obj)
                if item:
                    matches.append(item)
        else:
            # JSON lines
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                item = extract_text_and_rating(obj)
                if item:
                    matches.append(item)

    if len(matches) <= requested:
        return matches
    return random.sample(matches, requested)


def sample_categories(data_dir: str, categories: List[str], sample_size: int, out_dir: str = ".", out_prefix: str = "sampled_") -> Dict[str, int]:
    os.makedirs(out_dir, exist_ok=True)
    final_samples: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}

    for cat_file in categories:
        path = os.path.join(data_dir, cat_file)
        if not os.path.isabs(path):
            # assume relative to CWD if not absolute
            path = os.path.join(os.getcwd(), path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Category file not found: {path}")
        samples = sample_category_file(path, sample_size)
        counts[cat_file] = len(samples)
        for s in samples:
            s["category"] = os.path.splitext(cat_file)[0]
            final_samples.append(s)

    # Shuffle combined dataset
    random.shuffle(final_samples)

    # Write CSV and JSONL outputs
    csv_path = os.path.join(out_dir, f"{out_prefix}dataset.csv")
    jsonl_path = os.path.join(out_dir, f"{out_prefix}dataset.jsonl")

    import csv

    with open(csv_path, "w", encoding="utf-8", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["category", "rating", "review"])
        for row in final_samples:
            writer.writerow([row["category"], row["rating"], row["review"]])

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for row in final_samples:
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")

    return counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample reviews from category files")
    parser.add_argument("--data_dir", default="Data")
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--sample_size", type=int, default=10000)
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--out_prefix", default="sampled_")

    args = parser.parse_args()
    result = sample_categories(args.data_dir, args.categories, args.sample_size, args.out_dir, args.out_prefix)
    print("Sample counts:", result)
