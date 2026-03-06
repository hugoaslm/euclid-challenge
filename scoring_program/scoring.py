import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

EVAL_SETS = ["test", "private_test"]


def _clip_proba(p):
    p = np.asarray(p, dtype=float)
    return np.clip(p, 1e-15, 1.0 - 1e-15)


def _balanced_weights(y):
    """Balanced class weights computed on the current evaluation set."""
    y = y.astype(int)
    n = len(y)
    n1 = int(y.sum())
    n0 = n - n1
    w1 = n / (2.0 * max(n1, 1))
    w0 = n / (2.0 * max(n0, 1))
    return np.where(y == 1, w1, w0)


def weighted_log_loss(y, p):
    """Class-balanced weighted log loss on a single set."""
    y = np.asarray(y, dtype=int)
    p = _clip_proba(p)
    w = _balanced_weights(y)
    loss = -(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum()
    return float(loss)


def macro_redshift_weighted_log_loss(df, y_col = "y_quenched", p_col = "p_quenched", 
                                     zbin_col = "z_bin"):
    """
    Compute class-balanced weighted log loss per redshift bin,
    then average equally across bins (macro over bins).
    """
    bins = df[zbin_col].dropna().unique()
    if len(bins) == 0:
        raise ValueError("No redshift bins available for macro loss.")

    per_bin_losses = []
    for b in sorted(bins, key=str):
        g = df[df[zbin_col] == b]
        y = g[y_col].to_numpy(dtype=int)
        p = g[p_col].to_numpy(dtype=float)

        # If only one class appears in a bin, fall back to unweighted log loss in that bin
        if y.min() == y.max():
            p = _clip_proba(p)
            ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
            per_bin_losses.append(float(ll))
        else:
            per_bin_losses.append(weighted_log_loss(y, p))

    return float(np.mean(per_bin_losses))


def recall_at_precision(y, p, target_precision = 0.85):
    """Recall (completeness) achieved at or above a target precision (purity)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    order = np.argsort(-p)
    y_sorted = y[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y.sum()), 1)

    ok = precision >= target_precision
    if not np.any(ok):
        return 0.0
    return float(np.max(recall[ok]))


def _load_and_align(pred_path, ref_path):
    """
    Expect:
      - predictions: object_id, p_quenched  (probability of quenched, class=1)
      - reference:   object_id, y_quenched, z_bin
    """
    pred = pd.read_csv(pred_path)
    ref = pd.read_csv(ref_path)

    required_pred = {"object_id", "p_quenched"}
    required_ref = {"object_id", "y_quenched", "z_bin"}

    missing_pred = required_pred - set(pred.columns)
    missing_ref = required_ref - set(ref.columns)
    if missing_pred:
        raise ValueError(f"Predictions file missing columns: {sorted(missing_pred)}")
    if missing_ref:
        raise ValueError(f"Reference file missing columns: {sorted(missing_ref)}")

    pred = pred[["object_id", "p_quenched"]].copy()
    ref = ref[["object_id", "y_quenched", "z_bin"]].copy()

    pred["object_id"] = pred["object_id"].astype(str)
    ref["object_id"] = ref["object_id"].astype(str)
    ref["z_bin"] = ref["z_bin"].astype(str)

    merged = ref.merge(pred, on="object_id", how="left", validate="one_to_one")
    if merged["p_quenched"].isna().any():
        n_missing = int(merged["p_quenched"].isna().sum())
        raise ValueError(f"Missing predictions for {n_missing} object_id.")

    merged["p_quenched"] = merged["p_quenched"].astype(float)
    merged["y_quenched"] = merged["y_quenched"].astype(int)

    return merged


def main(reference_dir, prediction_dir, output_dir):
    scores = {}
    for eval_set in EVAL_SETS:
        print(f'Scoring {eval_set}')

        merged = _load_and_align(
            prediction_dir / f"{eval_set}_predictions.csv",
            reference_dir / f"{eval_set}_labels.csv",
        )

        y = merged["y_quenched"].to_numpy(dtype=int)
        p = merged["p_quenched"].to_numpy(dtype=float)

        # Primary: macro redshift class-balanced weighted log loss
        scores[f"{eval_set}_primary_wlogloss_macro_z"] = macro_redshift_weighted_log_loss(
            merged, y_col="y_quenched", p_col="p_quenched", zbin_col="z_bin"
        )

        # Secondary: AUPRC
        scores[f"{eval_set}_secondary_auprc"] = float(average_precision_score(y, p))

        # Tertiary: Recall at 85% precision
        scores[f"{eval_set}_tertiary_recall_at_p85"] = float(recall_at_precision(y, p, 0.85))

    # Add train and test times in the score
    json_durations = (prediction_dir / 'metadata.json').read_text()
    durations = json.loads(json_durations)
    scores.update(**durations)
    print(scores)

    # Write output scores
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'scores.json').write_text(json.dumps(scores))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scoring program for codabench"
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="/app/input/ref",
        help="",
    )
    parser.add_argument(
        "--prediction-dir",
        type=str,
        default="/app/input/res",
        help="",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/output",
        help="",
    )

    args = parser.parse_args()

    main(
        Path(args.reference_dir),
        Path(args.prediction_dir),
        Path(args.output_dir)
    )
