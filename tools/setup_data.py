import argparse
import shutil
from pathlib import Path

PHASE = "dev_phase"


def main():
    data_dir = Path(PHASE) / "input_data"
    ref_dir = Path(PHASE) / "reference_data"

    required = [
        data_dir / "train" / "train_features.csv",
        data_dir / "train" / "train_labels.csv",
        data_dir / "test" / "test_features.csv",
        data_dir / "private_test" / "private_test_features.csv",
        ref_dir / "test_labels.csv",
        ref_dir / "private_test_labels.csv",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required dev_phase files:\n  - " + "\n  - ".join(missing)
        )

    print("OK: dev_phase input_data/ and reference_data/ look ready.")


if __name__ == "__main__":
    main()