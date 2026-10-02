"""Turn text into dense embedding vectors on the Mac GPU (MPS) and cache them as .npy.

Usage:
  python 06_embed.py --model specter2 --text title
  python 06_embed.py --model qwen3-4b --text abstract

Rows follow the order of data/interim/clean.parquet. For --text abstract, papers without an
abstract get a zero vector, and a matching *_mask.npy marks which rows are real.

Models:
  minilm    all-MiniLM-L6-v2 (384-d)  - small general encoder, stand-in for Stanford's USE
  specter2  allenai/specter2 (768-d)  - trained on scientific papers and their citations
  qwen3-0.6b / qwen3-4b               - recent general-purpose embedding models
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from config import INTERIM, PROCESSED

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_texts(kind):
    clean = pd.read_parquet(INTERIM / "clean.parquet", columns=["paper_id", "title"])
    if kind == "title":
        return clean["title"].tolist(), np.ones(len(clean), dtype=bool)
    abstracts = pd.read_parquet(INTERIM / "abstracts.parquet").set_index("paper_id")["abstract"]
    text = clean["paper_id"].map(abstracts)
    mask = text.notna().to_numpy()
    return text.fillna("").tolist(), mask


def embed_sentence_transformers(name, texts, batch_size):
    from sentence_transformers import SentenceTransformer
    kwargs = {"model_kwargs": {"dtype": torch.float16}} if "Qwen" in name else {}
    model = SentenceTransformer(name, device=DEVICE, **kwargs)
    model.max_seq_length = 512
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=False)


def embed_specter2(texts, batch_size):
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
    model.to(DEVICE).eval()

    # Sort by length so each batch has similar lengths (less padding, much faster).
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            idx = order[i:i + batch_size]
            batch = tok([texts[j] for j in idx], padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(DEVICE)
            cls = model(**batch).last_hidden_state[:, 0, :]  # SPECTER2 uses the [CLS] vector
            out[idx] = cls.float().cpu().numpy()
            if (i // batch_size) % 200 == 0:
                print(f"  {i:,}/{len(texts):,}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["specter2", *MODELS])
    ap.add_argument("--text", required=True, choices=["title", "abstract"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, help="embed only the first N rows (speed test)")
    args = ap.parse_args()

    texts, mask = load_texts(args.text)
    if args.limit:
        texts, mask = texts[:args.limit], mask[:args.limit]
    todo = [t for t, m in zip(texts, mask) if m]
    print(f"{args.model} on {args.text}: {len(todo):,} texts, device={DEVICE}")

    start = time.time()
    if args.model == "specter2":
        vecs = embed_specter2(todo, args.batch_size)
    else:
        vecs = embed_sentence_transformers(MODELS[args.model], todo, args.batch_size)
    secs = time.time() - start
    print(f"Done in {secs / 60:.1f} min ({len(todo) / secs:.0f} texts/s), dim={vecs.shape[1]}")

    full = np.zeros((len(texts), vecs.shape[1]), dtype=np.float32)
    full[mask] = vecs
    if args.limit:
        return  # speed test only, nothing saved
    np.save(PROCESSED / f"emb_{args.model}_{args.text}.npy", full)
    if args.text == "abstract":
        np.save(PROCESSED / f"emb_{args.model}_{args.text}_mask.npy", mask)
    print(f"Saved emb_{args.model}_{args.text}.npy")


if __name__ == "__main__":
    main()
