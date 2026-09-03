import torch.optim as optim 
from tqdm import tqdm 
from src.models import ResNet18

import click
import torch
import torch.nn as nn

from src.data.dataset import get_dataloaders
from src.models.factory import get_model
from src.utils.config import load_config

@click.command()
@click.option(
    "--config",
    default="configs/resnet.yaml",
    help="Path to config file"
)
def main(config):

    # 1. Загрузить конфигурацию
    cfg = load_config(config)

    # 2. Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 3. Data
    train_loader, test_loader = get_dataloaders(
        batch_size=cfg["batch_size"]
    )

    # 4. Model
    model = get_model(cfg["model"]["type"])
    model = model.to(device)

    # 5. Loss
    loss_func = nn.CrossEntropyLoss()

    # 6. Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"]
    )

    # 7. Training
    
    epochs = cfg["training"]["epochs"]
    
    for _ in range(epochs):
        
        model.train()
        
        for X_train, y_train in tqdm(train_loader): 
            
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            
            predictions = model(X_train)
            
            loss = loss_func(predictions, y_train)
            
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

    # 8. Save model
    torch.save(
        model.state_dict(),
        f"checkpoints/{cfg['model']['type']}.pth"
    )

    print("Model saved.")

if __name__ == "__main__":
    main()