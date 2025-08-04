import torch
import torch.nn.functional as F
import torch.optim as optim
from utils.helpers import save_checkpoint


def train(model, train_loader, val_loader, cfg):
    device = cfg['device']
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{cfg['epochs']}], Loss: {avg_loss:.4f}")
        save_checkpoint(model, epoch, cfg)

    print("Training complete.")