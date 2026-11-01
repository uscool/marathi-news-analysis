"""Train the light headline classifier.

    python train_classifier.py --data data/marathi_headlines.csv
CSV schema: headline,label  (header optional)
"""
import argparse

from mna.classifier import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/classifier")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    info = train(a.data, a.out, hidden=a.hidden, test_size=a.test_size, seed=a.seed)
    print(f"labels: {info['labels']}  train={info['n_train']} test={info['n_test']}")
    print(info["report"])
    print(f"macro-F1: {info['macro_f1']:.3f}\nsaved to {info['path']}")
