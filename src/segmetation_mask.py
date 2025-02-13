import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms
import open_clip
from detectron2.config import get_cfg
from san.config import add_san_config 
from san.model.san import SAN
from detectron2.data import MetadataCatalog

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

# 1. Load SAN model with pre-trained weights
def load_san_model(checkpoint_path):
    # Initialize Detectron2 config
    cfg = get_cfg()
    add_san_config(cfg)  # Apply SAN-specific settings

    # Create SAN model properly
    model = SAN(cfg)  # Ensure it is an actual instance of SAN

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Check if checkpoint is returning a dictionary
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        raise ValueError("Checkpoint file does not contain 'model' key. Verify the file.")

    model.to(DEVICE)
    model.eval()

    return model, cfg 


# 2. Preprocess image for SAN
# def preprocess_image(image_path):
#     orig_image = Image.open(image_path).convert("RGB")
#     W, H = orig_image.size
#     transform = transforms.Compose([
#         transforms.Resize((640, 640)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
#     img_tensor = transform(orig_image).unsqueeze(0).to(DEVICE)
#     return img_tensor, (H, W), orig_image

def preprocess_image(image_path):
    orig_image = Image.open(image_path).convert("RGB")
    W, H = orig_image.size  # Store original size
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Match model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(orig_image).unsqueeze(0).to(DEVICE)
    return img_tensor, (H, W), orig_image  # Return (H, W) instead of (W, H)



# 3. Run SAN inference (get segmentation mask)
def get_segmentation_mask(model, image_tensor):
    with torch.no_grad():
        # Register a dummy dataset in MetadataCatalog
        dataset_name = "default"
        if dataset_name not in MetadataCatalog.list():
            MetadataCatalog.get(dataset_name).thing_classes = CLASSES
            MetadataCatalog.get(dataset_name).stuff_classes = CLASSES

        # Create batched input
        batched_inputs = [{
            "image": image_tensor.squeeze(0),
            "meta": {"dataset_name": dataset_name}  # Ensure meta key exists
        }]
        
        # Forward pass through SAN model
        outputs = model(batched_inputs)

        
        # Extract segmentation mask
        seg_mask = outputs[0]["sem_seg"].argmax(dim=0)
        print(seg_mask.shape)
        print(seg_mask)

    return seg_mask.cpu().numpy()


# 4. Visualization

# def visualize_results(image_path, seg_map):
#     orig_image = Image.open(image_path).convert("RGB")
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(orig_image)

#     # Overlay segmentation mask
#     ax.imshow(seg_map, alpha=0.5, cmap="jet")

#     plt.title("SAN Segmentation Result")
#     plt.axis("off")
#     plt.show()

def visualize_results(image_path, seg_map, orig_size):
    orig_image = Image.open(image_path).convert("RGB")
    # Correctly use (width, height) for resizing
    new_size = (orig_size[1], orig_size[0])  # (W, H)
    seg_map_resized = Image.fromarray(seg_map.astype(np.uint8)).resize(new_size, Image.NEAREST)
    seg_map_resized = np.array(seg_map_resized)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(orig_image)
    ax.imshow(seg_map_resized, alpha=0.5, cmap="jet")
    plt.title("SAN Segmentation Result")
    plt.axis("off")
    plt.show()


# Main execution
if __name__ == "__main__":
    # Load SAN model
    san_model, _ = load_san_model(CHECKPOINT_PATH)

    # Process image
    img_tensor, orig_size, orig_image = preprocess_image(IMAGE_PATH)

    # Get segmentation mask
    seg_map = get_segmentation_mask(san_model, img_tensor)

    # Visualize results
    visualize_results(IMAGE_PATH, seg_map, orig_size)
