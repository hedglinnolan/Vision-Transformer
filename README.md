# Vision Transformer (ViT) in PyTorch

This project contains a clean, modular implementation of a Vision Transformer for image classification (e.g., CIFAR-10).

## 📁 Project Structure
```
vision_transformer_project/
├── vit_example.py        # Main script for training/evaluation
├── config.py             # Configuration dictionary
├── requirements.txt      # Project dependencies
├── README.md             # Project overview and instructions
├── models/vit.py         # ViT model definition
├── data/dataset.py       # Data loading and transforms
├── train/trainer.py      # Training loop logic
├── train/evaluate.py     # Evaluation logic
├── utils/helpers.py      # Utilities (checkpointing, etc.)
├── utils/visualize.py    # Attention map visualization
└── checkpoints/          # Saved model weights
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python vit_example.py --mode train --epochs 20 --lr 1e-4
```

### 3. Evaluate the Model
```bash
python vit_example.py --mode eval --checkpoint checkpoints/vit_epoch10.pt
```

### 4. Visualize Attention
```python
from models.vit import VisionTransformer
from utils.visualize import visualize_attention
import torch

# Load model and image
model = VisionTransformer(**cfg['model_params']).to(cfg['device'])
model.load_state_dict(torch.load('checkpoints/vit_epoch10.pt', map_location=cfg['device']))
# Assume `image_tensor` is a torch.Tensor of shape [3, H, W]
visualize_attention(model, image_tensor)
```

## Theory

Below is the high-level theory behind our Vision Transformer:

1. **Patch Embedding**  
   Each input image is split into patches and linearly projected into embeddings…
   
2. **Self-Attention & Transformer Blocks**  
   The embeddings are processed via multi-head self-attention, followed by MLP layers…

---  
### Figures

<p align="center">
  <img src="docs/images/Embedding.jpeg" alt="Patch Embedding Diagram" width="400"/><br/>
  <em>Figure 1: How image patches are embedded into tokens.</em>
</p>

<p align="center">
  <img src="docs/images/Transforming.jpeg" alt="Self-Attention Flow" width="400"/><br/>
  <em>Figure 2: Overview of the Transformer block operations.</em>
</p>


## 📚 Features
- From-scratch ViT implementation
- Configurable via `config.py`
- CLI with `argparse`
- Checkpointing and logging utilities
- Attention visualization
- MPS/GPU/CPU support

## 🛠️ To Do
- Add custom dataset support
- Data augmentation pipelines
- Pretrained weights loading
- Experiment logging (TensorBoard, W&B)



---

## 🧪 Model Configuration Reference: Vision Transformer

This section lists **all configurable parameters and architectural components** in this Vision Transformer (ViT) pipeline. Use it as a guide for experimentation and model tuning.

---

### 📐 Input & Patching

| Parameter             | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| `image_size`          | Dimensions of input image (e.g., 224×224, 384×384)           |
| `patch_size`          | Size of each square patch (e.g., 16, 8, 32)                  |
| `in_channels`         | Number of image channels (3 for RGB, more for multispectral) |
| `flattened_patch_dim` | Implicit from `patch_size × patch_size × in_channels`        |

---

### 🧱 Model Architecture

| Parameter           | Description                                                     |
| ------------------- | --------------------------------------------------------------- |
| `embed_dim`         | Dimension of patch embeddings and attention vectors (e.g., 768) |
| `depth`             | Number of transformer encoder layers                            |
| `num_heads`         | Number of self-attention heads per layer                        |
| `mlp_dim`           | Hidden size of the feed-forward network (FFN) in each layer     |
| `dropout`           | Dropout applied globally (embeddings, MLP, etc.)                |
| `attention_dropout` | Dropout applied inside attention weights (if implemented)       |
| `cls_token`         | Classification token (can be replaced by pooling)               |
| `pos_embed`         | Positional encoding: learned or fixed (e.g., sinusoidal)        |
| `attention_type`    | Full, local, windowed, or sparse attention (for scalability)    |
| `normalization`     | `LayerNorm`, `BatchNorm`, or `RMSNorm` placement and type       |

---

### ⚙️ Training Configuration

| Parameter           | Description                                       |
| ------------------- | ------------------------------------------------- |
| `batch_size`        | Number of images per training step                |
| `epochs`            | Number of complete training passes                |
| `optimizer`         | Optimizer algorithm (e.g., AdamW, SGD, LAMB)      |
| `learning_rate`     | Initial learning rate                             |
| `weight_decay`      | L2 regularization coefficient                     |
| `scheduler`         | Learning rate decay method (e.g., cosine, warmup) |
| `gradient_clipping` | Max gradient norm to prevent explosion            |

---

### 📊 Loss Function & Label Setup

| Parameter       | Description                                                     |
| --------------- | --------------------------------------------------------------- |
| `loss_type`     | `CrossEntropy`, `Focal`, or label smoothing                     |
| `num_classes`   | Number of output classes                                        |
| `multi_label`   | If true, uses sigmoid + BCE loss for multilabel output          |
| `class_weights` | Custom weights for imbalanced classes (e.g., riot vs. non-riot) |

---

### 🗺️ Input Data Strategy

| Parameter            | Description                                      |
| -------------------- | ------------------------------------------------ |
| `crop_size`          | Size of image snippet (e.g., 672m × 672m)        |
| `patch_overlap`      | Whether overlapping patches are used             |
| `augmentations`      | Rotation, brightness, blur, jitter, etc.         |
| `normalization`      | Mean/std scaling per channel                     |
| `label_granularity`  | Image-level, patch-level, or pixel-level labels  |
| `riot_balance_ratio` | Percent of riot vs. non-riot samples in training (not yet implemented) |

---

### 🎯 Task Objective

| Parameter           | Description                                             |
| ------------------- | ------------------------------------------------------- |
| `output_mode`       | Classification, regression, segmentation, forecasting   |
| `auxiliary_heads`   | Optional heads: e.g., attention maps, region detection  |
| `multi_task`        | Predict multiple outputs (e.g., riot risk + crowd size) |
| `temporal_modeling` | Enable sequential data (e.g., ViViT, ConvLSTM)          |
| `metadata_inputs`   | Add ACLED, census, time, weather, etc.                  |

---

### 📈 Evaluation & Visualization

| Parameter        | Description                                |
| ---------------- | ------------------------------------------ |
| `metrics`        | Accuracy, AUC, precision, recall, F1       |
| `visualizations` | Attention maps, token heatmaps, saliency   |
| `explainability` | SHAP, LIME, Grad-CAM, Integrated Gradients |

---

### 🧠 Advanced Research Controls

| Parameter                | Description                                               |
| ------------------------ | --------------------------------------------------------- |
| `token_pruning`          | Prune low-importance tokens during inference              |
| `hierarchical_attention` | Swin-style local + global transformers                    |
| `shared_weights`         | Share encoder weights across layers (parameter-efficient) |
| `dynamic_depth`          | Early exit layers during inference                        |
| `prompt_tuning`          | Use learned text/image prompts (e.g., CLIP-style)         |
| `pretraining_strategy`   | From-scratch, supervised, or self-supervised (MAE, DINO)  |
