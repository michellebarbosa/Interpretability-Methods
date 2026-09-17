"""
Integrated Gradients for image classification.

Uses a pretrained ResNet18 (ImageNet) and Captum's IntegratedGradients
implementation to find which pixels most influenced the top prediction.
"""

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import urllib.request

from captum.attr import IntegratedGradients

# 1. Load pretrained model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels_path = "/tmp/imagenet_classes.txt"
urllib.request.urlretrieve(labels_url, labels_path)
with open(labels_path) as f:
    idx_to_label = [line.strip() for line in f.readlines()]

# 2. Preprocessing (must match ResNet's training preprocessing)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

# 3. Load a sample image (swap this for your own image path)
image_path = "/Users/michelle/Desktop/Interpretability-Methods/dog.jpg"
img = Image.open(image_path).convert("RGB")
img_np = np.array(img.resize((224, 224))) / 255.0  # [0,1], for display only

input_tensor = preprocess(img).unsqueeze(0)  # normalized tensor, for the model

# 4. Get the top prediction
with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.nn.functional.softmax(logits, dim=1)
top_idx = probs[0].argmax().item()
print(f"Top prediction: {idx_to_label[top_idx]} ({probs[0][top_idx]:.3f})")

# 5. Run Integrated Gradients
ig = IntegratedGradients(model)
baseline = torch.zeros_like(input_tensor)  # black image = "no information"

attributions, delta = ig.attribute(
    input_tensor,
    baselines=baseline,
    target=top_idx,
    n_steps=200,             # number of interpolation steps along the path
    return_convergence_delta=True,
)
# delta measures how well attributions satisfy the completeness property
# (should sum to output(real) - output(baseline)); near 0 is good.
print(f"Convergence delta (near 0 is good): {delta.item():.4f}")

# 6. Turn attributions into a viewable heatmap
# attributions shape: [1, 3, 224, 224] -> combine color channels
attr_np = attributions.squeeze().detach().numpy()
attr_map = np.abs(attr_np).sum(axis=0)
attr_map = attr_map / attr_map.max()  # normalize to [0, 1]

# 7. Plot original image + heatmap overlay
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img_np)
axes[0].set_title("Original image")
axes[0].axis("off")

axes[1].imshow(img_np)
axes[1].imshow(attr_map, cmap="hot", alpha=0.6)
axes[1].set_title(f"Integrated Gradients\n({idx_to_label[top_idx]})")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("/Users/michelle/Desktop/Interpretability-Methods/visualizations/integrated_gradients.png", bbox_inches="tight", dpi=150)
print("Saved heatmap to integrated_gradients.png")