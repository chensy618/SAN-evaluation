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
from san.model.clip_utils.utils import get_labelset_from_dataset

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
CHECKPOINT_PATH = "model/san_vit_b_16.pth"
IMAGE_PATH = "D:\\Github\\SAN-evaluation\\data\\images\\195.Carolina_Wren\\Carolina_Wren_0011_186871.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# 1. Load SAN model
# ------------------------------------------------------------------
def load_san_model(checkpoint_path):
    """Load SAN model with configuration"""
    cfg = get_cfg()
    add_san_config(cfg)
    model = SAN(cfg)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        raise ValueError("Invalid checkpoint format")
    
    model.to(DEVICE)
    model.eval()
    return model, cfg

# ------------------------------------------------------------------
# 2. Preprocess image
# ------------------------------------------------------------------
def preprocess_image(image_path):
    """Preprocess input image to 640x640 tensor"""
    orig_image = Image.open(image_path).convert("RGB")
    W, H = orig_image.size

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(orig_image).unsqueeze(0).to(DEVICE)
    return img_tensor, (H, W), orig_image

# ------------------------------------------------------------------
# 3. Process predictions from mask_logits
# ------------------------------------------------------------------
def process_predictions(mask_logits, orig_size):
    """Process model outputs and map to original image coordinates"""
    # Handle class number mismatch between model and dataset
    mask_logits = mask_logits[:, :-1].softmax(dim=1)  # [100, 16]
    class_ids = mask_logits.argmax(dim=1).cpu().numpy()
    
    # Get dataset labels (assuming no background class)
    dataset_classes = get_labelset_from_dataset("bird_parts")  # 15 classes
    num_dataset_classes = len(dataset_classes)
    
    # Add background class at beginning
    full_classes = ["background"] + dataset_classes
    
    # Validate class number alignment
    if mask_logits.shape[1] != len(full_classes):
        print(f"Notice: Model outputs {mask_logits.shape[1]} classes but dataset+background has {len(full_classes)}, applying automatic mapping")
    
    label_positions = []
    class_map = class_ids.reshape((10, 10))
    
    # Calculate original image parameters
    orig_height, orig_width = orig_size
    patch_w = orig_width // 10  # Patch width in original image
    patch_h = orig_height // 10  # Patch height in original image
    
    for i in range(10):
        for j in range(10):
            model_class_id = class_map[i, j]
            
            # Skip background (class 16 in model output)
            if model_class_id == 16:
                continue
                
            # Map model class ID to dataset labels
            dataset_class_id = model_class_id - 1
            
            # Validate mapping
            if dataset_class_id >= num_dataset_classes:
                print(f"Warning: Invalid class ID {model_class_id}, skipping")
                continue
                
            # Calculate center coordinates
            center_x = int((j * patch_w) + (patch_w / 2))
            center_y = int((i * patch_h) + (patch_h / 2))
            
            label_positions.append((
                (center_x, center_y),
                dataset_classes[dataset_class_id]
            ))
    
    return class_map, label_positions, (patch_h, patch_w)

# ------------------------------------------------------------------
# 4. Visualization
# ------------------------------------------------------------------
def visualize_results(image_path, class_map, orig_size, label_positions, patch_size):
    """Visualize results with segmentation overlay and labels"""
    orig_image = Image.open(image_path).convert("RGB")
    W, H = orig_image.size
    
    # Create colormap with background
    dataset_classes = get_labelset_from_dataset("bird_parts")
    full_classes = ["background"] + dataset_classes
    cmap = plt.cm.get_cmap("tab20", len(full_classes))
    
    # Generate segmentation map
    resized_class_map = np.kron(class_map, np.ones((patch_size[0], patch_size[1])))
    final_map = Image.fromarray(resized_class_map.astype(np.uint8)).resize(
        (W, H), 
        resample=Image.NEAREST
    )
    
    # Setup visualization
    plt.figure(figsize=(15, 10))
    ax = plt.gca()
    ax.imshow(orig_image)
    
    # Overlay segmentation (skip background)
    ax.imshow(final_map, alpha=0.3, cmap=cmap, vmin=0, vmax=len(full_classes)-1)
    
    # Plot labels with collision avoidance
    used_positions = set()
    for (x, y), label in sorted(label_positions, key=lambda x: -x[0][0]):
        offset = 0
        base_offset = 15
        while (int(x+offset), int(y+offset)) in used_positions and offset < 50:
            offset += base_offset
        
        used_positions.add((int(x+offset), int(y+offset)))
        
        ax.text(
            x + offset, y + offset, label,
            fontsize=9,
            weight='bold',
            color='black',
            va='center',
            ha='center',
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor='white',
                edgecolor='gray',
                alpha=0.8,
                linewidth=0.5
            )
        )
    
    # Create legend (excluding background)
    legend_elements = [
        plt.Line2D([0], [0], 
                  marker='s', 
                  color='w',
                  markerfacecolor=cmap(i+1),  # Skip background color
                  markersize=15,
                  label=label)
        for i, label in enumerate(dataset_classes)
    ]
    
    ax.legend(
        handles=legend_elements,
        title="Bird Parts",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False
    )
    
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load model
    model, cfg = load_san_model(CHECKPOINT_PATH)
    
    # 2. Preprocess image
    img_tensor, orig_size, orig_img = preprocess_image(IMAGE_PATH)
    
    # 3. Get predictions
    with torch.no_grad():
        dataset_name = "bird_parts"
        batched_inputs = [{"image": img_tensor.squeeze(0),
                           "meta": {"dataset_name": dataset_name}}]
        outputs = model(batched_inputs)
        mask_logits = outputs[0]["mask_logits"]
        print("mask_logits shape:", mask_logits.shape)
    
    # 4. Process predictions
    class_map, label_positions, patch_size = process_predictions(mask_logits, orig_size)
    
    # 5. Visualize
    visualize_results(IMAGE_PATH, class_map, orig_size, label_positions, patch_size)