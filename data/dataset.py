import torch
from torchvision import datasets, transforms


def get_dataloader(cfg):
    # Preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((cfg['model_params']['img_size'], cfg['model_params']['img_size'])),
        transforms.ToTensor(),
    ])

    # CIFAR-10 dataset
    train_ds = datasets.CIFAR10(root=cfg['data_path'], train=True, download=True, transform=transform)
    val_ds   = datasets.CIFAR10(root=cfg['data_path'], train=False, download=True, transform=transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=4
    )

    return train_loader, val_loader