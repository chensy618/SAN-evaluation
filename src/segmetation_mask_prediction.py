import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from detectron2.config import get_cfg
from san.config import add_san_config
from san.model.san import SAN
from detectron2.data import MetadataCatalog
import cv2

# Configuration
CHECKPOINT_PATH = "model/san_vit_b_16.pth"  # Pre-trained SAN model
IMAGE_PATH = "D:\\Github\\SAN-evaluation\\data\\images\\195.Carolina_Wren\\Carolina_Wren_0011_186871.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class labels
CLASSES = [
    "back", "beak", "belly", "breast", "crown", "forehead",
    "left eye", "left leg", "left wing", "nape", "right eye",
    "right leg", "right wing", "tail", "throat"
]

# 1. Load SAN model
def load_san_model(checkpoint_path):
    cfg = get_cfg()
    add_san_config(cfg)
    model = SAN(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        raise ValueError("Checkpoint file does not contain 'model' key. Verify the file.")

    model.to(DEVICE)
    model.eval()
    return model, cfg

# 2. Preprocess image
def preprocess_image(image_path):
    orig_image = Image.open(image_path).convert("RGB")
    W, H = orig_image.size
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(orig_image).unsqueeze(0).to(DEVICE)
    return img_tensor, (H, W), orig_image

# 3. Get segmentation mask
def get_segmentation_mask(model, image_tensor):
    with torch.no_grad():
        dataset_name = "default"
        if dataset_name not in MetadataCatalog.list():
            MetadataCatalog.get(dataset_name).thing_classes = CLASSES
            MetadataCatalog.get(dataset_name).stuff_classes = CLASSES

        batched_inputs = [{"image": image_tensor.squeeze(0), "meta": {"dataset_name": dataset_name}}]
        outputs = model(batched_inputs)
        seg_mask = outputs[0]["sem_seg"].argmax(dim=0)

    return seg_mask.cpu().numpy()

# 4. Extract labels from segmentation
def get_labels_from_segmentation(seg_map):
    """
    Extract class labels directly from the segmentation mask 
    and obtain the center point of each segmented region.
    """
    unique_labels = np.unique(seg_map)
    label_positions = {}
    print(seg_map.shape)
    
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
        
        mask = (seg_map == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Get all contours

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue  # Avoid division by zero

            # Compute the center point of the region
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Get class name
            class_name = CLASSES[label]

            # Allow multiple labels to be stored at the same coordinate
            if (cx, cy) not in label_positions:
                label_positions[(cx, cy)] = []
            label_positions[(cx, cy)].append(class_name)

    return label_positions


# 5. Visualization
def visualize_results(image_path, seg_map, orig_size, label_positions):
    orig_image = Image.open(image_path).convert("RGB")
    new_size = (orig_size[1], orig_size[0])  # (W, H)
    seg_map_resized = Image.fromarray(seg_map.astype(np.uint8)).resize(new_size, Image.NEAREST)
    seg_map_resized = np.array(seg_map_resized)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(orig_image)
    ax.imshow(seg_map_resized, alpha=0.5, cmap="jet")

    # Annotate class labels (supporting multiple labels)
    for (cx, cy), labels in label_positions.items():
        label_text = ", ".join(labels)  # Merge multiple labels into a single string
        ax.text(cx, cy, label_text, color="black", fontsize=12,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor='none'))

    ax.set_title("SAN Segmentation Labels")
    ax.axis("off")
    plt.show()


# Main execution
if __name__ == "__main__":
    san_model, _ = load_san_model(CHECKPOINT_PATH)
    img_tensor, orig_size, orig_image = preprocess_image(IMAGE_PATH)
    seg_map = get_segmentation_mask(san_model, img_tensor)

    # Use SAN-generated labels directly
    label_positions = get_labels_from_segmentation(seg_map)

    # Visualize the final result
    visualize_results(IMAGE_PATH, seg_map, orig_size, label_positions)
