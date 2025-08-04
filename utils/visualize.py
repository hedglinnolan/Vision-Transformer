import os
import sys
# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from config import cfg
from models.vit import VisionTransformer

def compute_attention_map(model, image_tensor, layer_idx=0):
    """Compute the CLS-to-patch attention map for a transformer layer."""
    model.eval()
    device = next(model.parameters()).device
    img = image_tensor.to(device)

    # Patch embedding + CLS token + positional embedding
    x = model.patch_embed(img.unsqueeze(0))  # (1, N, E)
    B, N, E = x.shape
    cls_tok = model.cls_token.expand(B, -1, -1)  # (1, 1, E)
    x = torch.cat((cls_tok, x), dim=1)           # (1, 1+N, E)
    x = x + model.pos_embed[:, : x.size(1), :]

    # Self-attention on specified layer
    _, attn_weights = model.encoder.layers[layer_idx].self_attn(
        x, x, x,
        need_weights=True,
        average_attn_weights=False
    )  # (1, heads, seq_len, seq_len)

    # Extract CLS→patch attention
    cls_attn = attn_weights[0, :, 0, 1:]   # (heads, N)
    cls_attn = cls_attn.mean(0)           # (N,)
    
    grid_size = int(np.sqrt(cls_attn.shape[0]))
    attn_map = cls_attn.detach().reshape(grid_size, grid_size).cpu().numpy()
    return attn_map

def show_original_image(image_tensor, label=None):
    """Display the original image."""
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    plt.imshow(img)
    if label is not None:
        plt.title(f"Original Image (class={label})")
    plt.axis('off')
    plt.show()

def plot_paper_figure(image_tensor, attn_map):
    """Create a 3-panel figure for a paper: original, heatmap, overlay."""
    patch_size = cfg['model_params']['patch_size']
    heatmap_upsampled = np.kron(attn_map, np.ones((patch_size, patch_size)))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 1: original
    axes[0].imshow(image_tensor.cpu().numpy().transpose(1,2,0))
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Panel 2: standalone heatmap
    im = axes[1].imshow(attn_map, cmap='viridis', interpolation='nearest')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    # Panel 3: overlay
    axes[2].imshow(image_tensor.cpu().numpy().transpose(1,2,0), alpha=1.0)
    axes[2].imshow(heatmap_upsampled, cmap='viridis', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Paper-style ViT Attention Visualization")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--index', type=int, default=0, help='Validation set index')
    parser.add_argument('--layer', type=int, default=0, help='Transformer layer index')
    args = parser.parse_args()

    # Load model
    model = VisionTransformer(**cfg['model_params']).to(cfg['device'])
    state = torch.load(args.checkpoint, map_location=cfg['device'])
    model.load_state_dict(state)

    # Load image
    transform = transforms.Compose([
        transforms.Resize((cfg['model_params']['img_size'], cfg['model_params']['img_size'])),
        transforms.ToTensor(),
    ])
    val_ds = datasets.CIFAR10(root=cfg['data_path'], train=False, download=True, transform=transform)
    image_tensor, label = val_ds[args.index]

    # Generate and plot
    attn_map = compute_attention_map(model, image_tensor, layer_idx=args.layer)
    show_original_image(image_tensor, label)
    plot_paper_figure(image_tensor, attn_map)
