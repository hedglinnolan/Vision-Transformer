import argparse
import torch
from config import cfg
from models.vit import VisionTransformer
from data.dataset import get_dataloader
from train.trainer import train
from train.evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate a Vision Transformer.")
    parser.add_argument('--mode', type=str, default=cfg['mode'], choices=['train', 'eval'],
                        help='Operation mode: train or eval')
    parser.add_argument('--epochs', type=int, default=cfg['epochs'],
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=cfg['lr'],
                        help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for evaluation or resuming')
    return parser.parse_args()


def main():
    args = parse_args()
    # Update config
    cfg['mode'] = args.mode
    cfg['epochs'] = args.epochs
    cfg['lr'] = args.lr

    # Initialize model and move to device
    model = VisionTransformer(**cfg['model_params']).to(cfg['device'])

    # Load checkpoint if provided
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=cfg['device'])
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")

    # Data loaders
    train_loader, val_loader = get_dataloader(cfg)

    # Train or evaluate
    if cfg['mode'] == 'train':
        train(model, train_loader, val_loader, cfg)
    else:
        evaluate(model, val_loader, cfg)


if __name__ == '__main__':
    main()