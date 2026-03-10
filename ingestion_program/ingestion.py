import json
import sys
import time
from pathlib import Path

import pandas as pd


def load_train(data_dir: Path):
    train_dir = data_dir / "train"

    X = pd.read_csv(train_dir / "train_features.csv", dtype={"object_id": "string"})
    y = pd.read_csv(train_dir / "train_labels.csv", dtype={"object_id": "string"})

    y = y[["object_id", "y_quenched"]].copy()
    X["object_id"] = X["object_id"].astype("string")
    y["object_id"] = y["object_id"].astype("string")

    df = X.merge(y, on="object_id", how="inner", validate="one_to_one")

    X_train = df.drop(columns=["object_id", "y_quenched"])
    y_train = df["y_quenched"].astype(int)

    return X_train, y_train


def load_test(data_dir: Path, feature_cols):
    test_path = data_dir / "test" / "test_features.csv"
    X_test = pd.read_csv(test_path, dtype={"object_id": "string"})
    X_test["object_id"] = X_test["object_id"].astype("string")

    # enforce training column order
    return X_test[["object_id", *feature_cols]]


def predict_with_ids(model, X_df: pd.DataFrame) -> pd.DataFrame:
    X_df = X_df.copy()
    obj_id = X_df["object_id"].astype(str)
    X = X_df.drop(columns=["object_id"])

    p_quenched = model.predict_proba(X)[:, 1].astype(float)
    return pd.DataFrame({
        "object_id": obj_id.values,
        "p_quenched": p_quenched,
    })


def main(data_dir, output_dir):
    from submission import get_model

    X_train, y_train = load_train(data_dir)
    feature_cols = list(X_train.columns)

    print("Training the model")
    model = get_model()

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    print("-" * 10)
    print("Evaluate the model")

    start = time.time()
    X_test = load_test(data_dir, feature_cols)
    predictions = predict_with_ids(model, X_test)
    test_time = time.time() - start

    duration = train_time + test_time
    print("-" * 10)
    print(f"Completed Prediction. Total duration: {duration}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metadata.json", "w+") as f:
        json.dump(
            dict(train_time=train_time, test_time=test_time, duration=duration),
            f
        )

    predictions.to_csv(output_dir / "predictions.csv", index=False)

    print()
    print("Ingestion Program finished. Moving on to scoring")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingestion program for codabench")
    parser.add_argument("--data-dir", type=str, default="/app/input_data")
    parser.add_argument("--output-dir", type=str, default="/app/output")
    parser.add_argument("--submission-dir", type=str, default="/app/ingested_program")

    args = parser.parse_args()
    sys.path.append(args.submission_dir)
    sys.path.append(str(Path(__file__).parent.resolve()))

    main(Path(args.data_dir), Path(args.output_dir))