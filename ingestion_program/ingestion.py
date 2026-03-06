import json
import sys
import time
from pathlib import Path

import pandas as pd


EVAL_SETS = ["test", "private_test"]


def load_train(data_dir: Path):
    train_dir = data_dir / "train"

    X = pd.read_csv(train_dir / "train_features.csv", dtype={"object_id": "string"})
    y = pd.read_csv(train_dir / "train_labels.csv", dtype={"object_id": "string"})

    X = X.copy()
    y = y[["object_id", "y_quenched"]].copy()

    X["object_id"] = X["object_id"].astype("string")
    y["object_id"] = y["object_id"].astype("string")

    # Align
    df = X.merge(y, on="object_id", how="inner", validate="one_to_one")

    X_train = df.drop(columns=["object_id", "y_quenched"])
    y_train = df["y_quenched"].astype(int)

    return X_train, y_train


def predict_with_ids(model, X_df: pd.DataFrame) -> pd.DataFrame:
    X_df = X_df.copy()
    X_df["object_id"] = X_df["object_id"].astype("string")

    obj_id = X_df["object_id"]
    X = X_df.drop(columns=["object_id"])

    p_quenched = model.predict_proba(X)[:, 1].astype(float)
    return pd.DataFrame({"object_id": obj_id.astype(str).values, "p_quenched": p_quenched})


def main(data_dir, output_dir):
    # Here, you can import info from the submission module, to evaluate the
    # submission
    from submission import get_model

    X_train, y_train = load_train(data_dir)

    print("Training the model")

    model = get_model()

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    print("-" * 10)
    print("Evaluate the model")

    start = time.time()
    res = {}
    for eval_set in EVAL_SETS:
        X_test = pd.read_csv(data_dir / eval_set / f"{eval_set}_features.csv")
        res[eval_set] = predict_with_ids(model, X_test)

    test_time = time.time() - start
    print("-" * 10)
    duration = train_time + test_time
    print(f"Completed Prediction. Total duration: {duration}")

    # Write output files
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metadata.json", "w+") as f:
        json.dump(dict(train_time=train_time, test_time=test_time), f)

    for eval_set in EVAL_SETS:
        filepath = output_dir / f"{eval_set}_predictions.csv"
        res[eval_set].to_csv(filepath, index=False)

    print()
    print("Ingestion Program finished. Moving on to scoring")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingestion program for codabench"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/app/input_data",
        help="",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/output",
        help="",
    )
    parser.add_argument(
        "--submission-dir",
        type=str,
        default="/app/ingested_program",
        help="",
    )

    args = parser.parse_args()
    sys.path.append(args.submission_dir)
    sys.path.append(Path(__file__).parent.resolve())

    main(Path(args.data_dir), Path(args.output_dir))
