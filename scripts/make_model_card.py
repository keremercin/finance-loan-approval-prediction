import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = Path("reports/metrics.csv")
    if not p.exists():
        raise SystemExit("Run training first: python scripts/train_model.py")

    df = pd.read_csv(p)
    top = df.iloc[0]
    metadata_path = Path("reports/metrics.json")
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )

    lines = [
        "# Model Card — Loan Approval Prediction",
        "",
        f"Best model: **{top['model']}**",
        "",
        f"Application version: **{metadata.get('app_version', 'unknown')}**",
        "",
        metadata.get("evaluation_method", "Evaluation method not recorded."),
        "",
        (
            "The API score is uncalibrated. This is not validated for production or "
            "real credit decisions. The repository does not identify the dataset "
            "source or license; `data/test.csv` has no target label and is not used "
            "label and is not used for testing."
        ),
        "",
        "## Validation Metrics",
        "",
        "| Model | Accuracy | F1 | ROC-AUC |",
        "|---|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['model']} | {r['accuracy']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} |"
        )

    out = Path("reports/model_card.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("written", out)


if __name__ == "__main__":
    main()
