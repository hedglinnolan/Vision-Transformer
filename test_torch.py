from ViT_example import *

model = VisionTransformer(img_size=224, patch_size=16, num_classes=10)
dummy_img = torch.randn(1, 3, 224, 224)  # Batch of 1, 3 channels, 224x224
logits = model(dummy_img)
print(logits.shape)  # Should be [1, 10]
