import click
import torch

from src.data.dataset import get_dataloaders
from src.models.factory import get_model
from src.utils.config import load_config

from src.utils.logger import log_evaluation

@click.command()
@click.option(
    "--config",
    default="configs/resnet18.yaml",
    help="Path to config file"
)
def main(config):

    # 1. Load config
    cfg = load_config(config)

    # 2. Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 3. Data
    _, test_loader = get_dataloaders(
        batch_size=cfg["training"]["batch_size"],
        augmentation_config=cfg["augmentation"],
    )

    # 4. Model
    model = get_model(cfg["model"]["type"])

    checkpoint_path = (
        f"./src/models/checkpoints/{cfg['model']['type']}.pth"
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    model = model.to(device)

    # 5. Evaluation mode
    model.eval()

    # 6. Evaluation
    correct = 0
    total = 0

    with torch.no_grad():

        for X_test, y_test in test_loader:

            X_test = X_test.to(device)
            y_test = y_test.to(device)

            predictions = model(X_test)

            predicted_classes = predictions.argmax(dim=1)

            total += y_test.size(0)
            correct += (
                predicted_classes == y_test
            ).sum().item()

    # 7. Accuracy
    accuracy = correct / total

    print(f"Test accuracy: {accuracy:.4f}")

    log_evaluation(cfg, accuracy)

if __name__ == "__main__":
    main()