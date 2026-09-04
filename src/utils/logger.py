import csv
from pathlib import Path
from datetime import datetime

def log_evaluation(cfg, accuracy):
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "evaluation.csv"

    file_exists = log_file.exists()

    with open(log_file, "a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "model",
                "epochs",
                "batch_size",
                "learning_rate",
                "augmentation",
                "horizontal_flip",
                "random_crop",
                "rotation",
                "accuracy",
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cfg["model"]["type"],
            cfg["training"]["epochs"],
            cfg["training"]["batch_size"],
            cfg["training"]["learning_rate"],
            cfg["augmentation"]["enabled"],
            cfg["augmentation"]["horizontal_flip"]["enabled"],
            cfg["augmentation"]["random_crop"]["enabled"],
            cfg["augmentation"]["rotation"]["enabled"],
            accuracy,
        ])