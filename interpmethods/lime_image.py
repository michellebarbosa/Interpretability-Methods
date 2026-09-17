"""
LIME for image classification.

Uses a pretrained ResNet18 (ImageNet) as the black-box model, and asks:
"which superpixels (regions) of the image drove the top prediction?"
"""

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from skimage.segmentation import mark_boundaries
from lime import lime_image
import matplotlib.pyplot as plt
import urllib.request

# 1. Load a pretrained model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# ImageNet class labels
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels_path = "/tmp/imagenet_classes.txt"
urllib.request.urlretrieve(labels_url, labels_path)
with open(labels_path) as f:
    idx_to_label = [line.strip() for line in f.readlines()]

# 2. Preprocessing pipeline (must match what ResNet expects)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

# 3. Load an example image (swap this path for your own image)
image_path = "/Users/michelle/Desktop/Interpretability-Methods/dog.jpg"
img = Image.open(image_path).convert("RGB")
img_np = np.array(img.resize((224, 224))) / 255.0  # for LIME, values in [0,1]

# 4. Prediction function LIME will call.
# Takes a batch of images (as numpy arrays, shape [N, H, W, 3]) and returns
# class probabilities, shape [N, num_classes].
def predict_fn(images):
    batch = torch.stack([preprocess(Image.fromarray((im * 255).astype(np.uint8)))
                          for im in images])
    with torch.no_grad():
        logits = model(batch)
        probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.numpy()

# 5. Check the top prediction for the original image
probs = predict_fn(np.expand_dims(img_np, axis=0))
top_idx = probs[0].argmax()
print(f"Top prediction: {idx_to_label[top_idx]} ({probs[0][top_idx]:.3f})")

# 6. Run LIME
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    img_np,
    predict_fn,
    top_labels=1,
    hide_color=0,
    num_samples=1000,  # number of perturbed images LIME generates internally
)

# 7. Visualize: highlight the superpixels supporting the top prediction
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=5,
    hide_rest=False,
)

plt.figure(figsize=(6, 6))
plt.imshow(mark_boundaries(temp, mask))
plt.title(f"Regions supporting: {idx_to_label[top_idx]}")
plt.axis("off")
plt.savefig("/Users/michelle/Desktop/Interpretability-Methods/visualizations/lime_image_explanation.png", bbox_inches="tight")
print("Saved visual explanation to lime_image_explanation.png")