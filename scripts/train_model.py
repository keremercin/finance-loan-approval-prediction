from loan_approval.train import train_and_evaluate


if __name__ == "__main__":
    result = train_and_evaluate()
    print("Best model:", result["best_model"])
    print("Best F1:", round(result["best_f1"], 4))
    print("Metrics saved:", result["metrics_path"])
    print("Model saved:", result["model_path"])
