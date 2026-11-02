"""Helper: build a predictions file from ground truth.

Used for validation/sanity checks. With ``--jitter`` it perturbs boundaries by a
fixed offset (seconds) so metrics degrade in a predictable way.
"""

import argparse
import json

from nyra_timing_benchmark import load_ground_truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", default="/home/ubuntu/data/english_speech")
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--jitter", type=float, default=0.0, help="Seconds to add to boundaries.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt = load_ground_truth(args.dataset, args.data_root)
    keys = list(gt.keys())
    if args.num_samples:
        keys = keys[: args.num_samples]

    predictions = {}
    for k in keys:
        sample = gt[k]
        entries = [
            {"word": w, "start": s + args.jitter, "end": e + args.jitter}
            for w, (s, e) in zip(sample.words, sample.timings)
        ]
        predictions[k] = entries

    with open(args.out, "w") as f:
        json.dump(
            {"model": f"gt_jitter{args.jitter}", "dataset": args.dataset, "predictions": predictions},
            f,
        )
    print(f"Wrote {len(predictions)} samples to {args.out}")


if __name__ == "__main__":
    main()
