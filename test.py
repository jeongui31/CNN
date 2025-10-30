#!/usr/bin/env python3
import argparse
from gensim.models import KeyedVectors
import numpy as np

def knn_cosine(vec, M, topk):
    v = vec / (np.linalg.norm(vec) + 1e-12)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    sims = Mn @ v
    idx = np.argpartition(-sims, range(min(topk, sims.size)))[:topk]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--tokens", default="!,," )
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    kv = KeyedVectors.load_word2vec_format(args.emb, binary=True)
    vocab = kv.index_to_key
    M = kv.vectors  # (V, D)
    k = args.k
    for q in [t for t in args.tokens.split(",") if t]:
        if q not in kv:
            print(f"[MISS] {repr(q)} not in vocab"); continue
        qi = kv.key_to_index[q]
        qv = kv[q]
        idx, sims = knn_cosine(qv, M, k+1)
        idx = [i for i in idx if i != qi][:k]
        print(f"\n=== {repr(q)} ===")
        for r,(i,s) in enumerate(zip(idx, sims[:len(idx)]),1):
            print(f"{r:2d}. {vocab[i]!r:20s}  sim={s:.6f}")

if __name__ == "__main__":
    main()
