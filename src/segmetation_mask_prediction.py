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
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.colormap import random_color
from san.model.clip_utils.utils import get_labelset_from_dataset

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
CHECKPOINT_PATH = "model/san_vit_b_16.pth"  # Path to your pre-trained SAN model
IMAGE_PATH = "D:\\Github\\SAN-evaluation\\data\\images\\195.Carolina_Wren\\Carolina_Wren_0011_186871.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# 1. Load SAN model
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# 2. Preprocess image (resize to 640x640)
# ------------------------------------------------------------------
def preprocess_image(image_path):
    orig_image = Image.open(image_path).convert("RGB")
    W, H = orig_image.size

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        # transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(orig_image).unsqueeze(0).to(DEVICE)
    return img_tensor, (H, W), orig_image


# ------------------------------------------------------------------
# 3. Get segmentation mask from SAN
# ------------------------------------------------------------------
def get_segmentation_mask(model, image_tensor):
    """
    Runs the model and returns the argmax segmentation mask (640x640).
    """
    with torch.no_grad():
        dataset_name = "bird_parts"
        batched_inputs = [{"image": image_tensor.squeeze(0),
                           "meta": {"dataset_name": dataset_name}}]
        outputs = model(batched_inputs)
        # outputs[0]["sem_seg"] is [num_classes, H, W]
        seg_mask = outputs[0]["sem_seg"].argmax(dim=0)
        print("Segmentation mask:", seg_mask)
        
    return seg_mask.cpu().numpy()


# ------------------------------------------------------------------
# 4. Merge disconnected regions for each label into a single bounding box
# ------------------------------------------------------------------
def get_merged_labels_from_segmentation(seg_map, orig_size):
    """
    - Resizes seg_map (which is 640x640) back to (orig_width x orig_height).
    - For each unique label, finds all pixels, merges them into one bounding box,
      and returns (cx, cy) -> label_name.
    """
    labelset = get_labelset_from_dataset("bird_parts")
    print("Label set:", labelset)

    unique_labels = np.unique(seg_map)
    print("Unique labels in seg_map:", unique_labels)

    # Resize seg_map back to original image size
    seg_map_resized = cv2.resize(seg_map,(orig_size[1], orig_size[0]),interpolation=cv2.INTER_NEAREST)

    label_positions = {}

    for label_id in unique_labels:
        # if label_id == 16:  # if skip background
        #     continue

        # Gather all pixels belonging to this label
        mask = (seg_map_resized == label_id)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue  # no pixels found for this label

        # Compute a single bounding box that encloses all pixels of this label
        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)
        w = max_x - min_x + 1
        h = max_y - min_y + 1

        # Center point of that bounding box
        cx = min_x + w // 2
        cy = min_y + h // 2

        # Get class name from labelset (or "Unknown_x" if out of range)
        if 0 <= label_id < len(labelset):
            class_name = labelset[label_id]
        else:
            class_name = f"Unknown_{label_id}"

        label_positions[(cx, cy)] = class_name

        print(f"Label {label_id} ('{class_name}') -> bounding box center: "
              f"({cx}, {cy}), size: ({w}, {h})")

    return label_positions


# ------------------------------------------------------------------
# 5. Visualization
# ------------------------------------------------------------------
def visualize_results(image_path, seg_map, orig_size, label_positions):
    """
    Overlays the segmentation (with transparency) on the original image.
    Draws one text label at the bounding box center for each label.
    """
    # Load the original image
    orig_image = Image.open(image_path).convert("RGB")
    new_size = (orig_size[1], orig_size[0])  # (width, height)

    # Resize seg_map (640x640) to the original image size
    resample_method = Image.NEAREST

    seg_map_resized = Image.fromarray(seg_map.astype(np.uint8)).resize(
        new_size,
        resample=resample_method
    )
    seg_map_resized = np.array(seg_map_resized)

    # Register metadata for "bird_parts" so we can visualize with a color map
    labelset = get_labelset_from_dataset("bird_parts")
    MetadataCatalog.get("bird_parts").set(thing_classes=labelset,
                                          stuff_classes=labelset)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(orig_image)

    # Create a color map with as many colors as we have classes
    cmap = plt.cm.get_cmap("jet", len(labelset))

    # Overlay the segmentation map
    ax.imshow(seg_map_resized, alpha=0.4, cmap=cmap, vmin=0, vmax=len(labelset) - 1)

    # Draw each bounding box center with text
    used_positions = set()
    for (cx, cy), label in sorted(label_positions.items(), key=lambda x: -x[0][0]):
        offset = 0
        # If there's overlap, shift the text slightly
        while (cx + offset, cy + offset) in used_positions:
            offset += 20
        used_positions.add((cx + offset, cy + offset))

        ax.text(cx + offset, cy + offset, label, color="black", fontsize=10, fontweight="bold", va="center", ha="center",
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="gray",
                    linewidth=0.5)
                )

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = []
    seg_labels_in_image = np.unique(seg_map_resized)
    for label_id in seg_labels_in_image:
        if 0 <= label_id < len(labelset):
            legend_elements.append(Patch(facecolor=cmap(label_id),label=labelset[label_id]))
    ax.legend(handles=legend_elements,title="Categories",loc="upper right",bbox_to_anchor=(1.25, 1))

    ax.set_title("SAN Segmentation Labels", fontsize=14, pad=20)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load model
    san_model, _ = load_san_model(CHECKPOINT_PATH)

    # 2. Preprocess image (to 640x640)
    img_tensor, orig_size, orig_image = preprocess_image(IMAGE_PATH)

    # 3. Get segmentation mask (640x640)
    seg_map = get_segmentation_mask(san_model, img_tensor)
    # print("seg_map :", seg_map)

    # 4. Merge bounding boxes for each label
    label_positions = get_merged_labels_from_segmentation(seg_map, orig_size)

    # 5. Visualize results
    visualize_results(IMAGE_PATH, seg_map, orig_size, label_positions)
