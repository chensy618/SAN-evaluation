import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import cv2
from detectron2.config import get_cfg
from san.config import add_san_config
from san.model.san import SAN
from san.model.clip_utils.utils import get_labelset_from_dataset

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
CHECKPOINT_PATH = "model/san_vit_b_16.pth"
IMAGE_PATH = "D:\\Github\\SAN-evaluation\\data\\images\\195.Carolina_Wren\\Carolina_Wren_0011_186871.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# 1. Load SAN Model
# ------------------------------------------------------------------
def load_san_model(checkpoint_path):
    cfg = get_cfg()
    add_san_config(cfg)
    model = SAN(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        raise ValueError("Invalid checkpoint format")

    model.to(DEVICE).eval()
    return model, cfg

# ------------------------------------------------------------------
# 2. Preprocess Image
# ------------------------------------------------------------------
def preprocess_image(image_path):
    orig_image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(orig_image).unsqueeze(0).to(DEVICE)
    return img_tensor, orig_image

# ------------------------------------------------------------------
# 3. Compute `16×16` Patch Labels (Removing Background)
# ------------------------------------------------------------------
def process_predictions(mask_preds, mask_cls_result):
    """
    Computes class labels for `16x16` pixel patches:
    - The 640x640 image consists of `40x40` patches.
    - Each patch gets the class with the **highest probability**.
    - **Removes background class**.

    Args:
    mask_preds: [1, 100, 40, 40] → Segmentation masks
    mask_cls_result: [100, 16] → Class predictions

    Returns:
    patch_labels: [40, 40] → The predicted class for each `16x16` pixel region
    """
    mask_preds = torch.sigmoid(mask_preds).squeeze(0)  # [100, 40, 40]
    mask_cls = F.softmax(mask_cls_result, dim=1)       # [100, 16]
    
    # Compute class activations (16, 40, 40)
    S = torch.einsum("nhw,cn->chw", mask_preds, mask_cls.T)  # [16, 40, 40]
    
    # **Remove background class** (last index is background)
    S_no_bg = S[:-1, :, :]  # Shape becomes [15, 40, 40]
    
    # **Compute the most probable class for each `16×16` pixel patch**
    patch_labels = S_no_bg.argmax(dim=0)  # Shape: [40, 40]

    print("Final patch labels shape:", patch_labels.shape)
    return patch_labels.cpu().numpy()

# ------------------------------------------------------------------
# 4. Patch-Level Visualization (Using `mpatches.Rectangle`)
# ------------------------------------------------------------------
def visualize_patches(image_array, patch_labels, label_names, alpha=0.5):
    """
    Visualizes segmentation for `16x16` pixel patches:
    - **Overlays semi-transparent color for each patch**
    - **Places class labels at the center of each patch**
    - **Each patch covers `16x16` pixels in the `640x640` image**

    Args:
    image_array: NumPy array of the original image
    patch_labels: [40, 40] NumPy array of class labels
    label_names: List of class names
    alpha: Transparency for overlay (0-1)
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image_array)

    patch_size = 16  # Since we have 40×40 patches but a 640×640 image, each patch spans 16 pixels
    num_patches = patch_labels.shape[0]  # 40

    # Loop over each `16×16` patch
    for i in range(num_patches):
        for j in range(num_patches):
            label_idx = patch_labels[i, j]  # Class index
            label = label_names[label_idx] if label_idx < len(label_names) else str(label_idx)

            x, y = j * patch_size, i * patch_size  # Patch position
            
            # Draw Patch Border
            rect = mpatches.Rectangle(
                (x, y), patch_size, patch_size,
                linewidth=1.0, edgecolor='white', facecolor='none'
            )
            ax.add_patch(rect)

            # Add Class Label
            ax.text(
                x + patch_size / 2, y + patch_size / 2, label,
                color='blue', fontsize=5, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
            )

    ax.set_title("40x40 Patch Segmentation (Each Patch is 16x16)")
    ax.axis('off')
    plt.show()

# ------------------------------------------------------------------
# 5. Main Execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    model, cfg = load_san_model(CHECKPOINT_PATH)
    img_tensor, orig_img = preprocess_image(IMAGE_PATH)

    with torch.no_grad():
        outputs = model([{"image": img_tensor.squeeze(0), "meta": {"dataset_name": "bird_parts"}}])
        mask_preds = outputs[0]["mask_preds"]    
        mask_cls_result = outputs[1]["mask_cls_result"]  

    patch_labels = process_predictions(mask_preds, mask_cls_result)

    label_names = get_labelset_from_dataset("bird_parts")

    image_array = np.array(orig_img.resize((640, 640)))

    visualize_patches(image_array, patch_labels, label_names, alpha=0.5)
