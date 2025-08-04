import os
import torch

def save_checkpoint(model, epoch, cfg):
    path = cfg.get('checkpoint_path', 'checkpoints')
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"vit_epoch{epoch+1}.pt")
    torch.save(model.state_dict(), filename)
    print(f"Saved checkpoint: {filename}")