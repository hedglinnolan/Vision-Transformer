import torch

cfg = {
    # Mode: 'train' or 'eval'
    'mode': 'train',
    # Device: MPS (Apple), CUDA (NVIDIA), or CPU
    'device': 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'),

    # Vision Transformer parameters
    'model_params': {
        'img_size': 224,
        'patch_size': 16,
        'in_channels': 3,
        'num_classes': 10,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_dim': 3072,
        'dropout': 0.1,
    },

    # Data and training settings
    'data_path': './data',
    'batch_size': 32,
    'epochs': 10,
    'lr': 3e-4,
}