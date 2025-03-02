import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms

#  Configuration 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "model/san_vit_b_16.pth"
IMAGE_PATH = "D:\\Github\\SAN-evaluation\\data\\images\\195.Carolina_Wren\\Carolina_Wren_0011_186871.jpg"

#  Model  
from san.model.san import SAN
from detectron2.config import get_cfg
from san.config import add_san_config

def load_san_model(checkpoint_path):
    """Load SAN model"""
    cfg = get_cfg()
    add_san_config(cfg)
    model = SAN(cfg)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    model.to(DEVICE)
    model.eval()
    return model, cfg

#  Data Processing  
def preprocess_image(image_path):
    """Image preprocessing pipeline"""
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711))
    ])
    return transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)

#  Attention Processing  
def process_attention_maps(attn_maps_tuple):
    """Process multi-head attention maps:
    1. Average across attention layers
    2. Average across attention heads
    3. Convert to per-query patch distributions"""
    # [num_layers, batch_size, num_heads, num_queries, num_patches]
    stacked_attn = torch.stack(attn_maps_tuple)
    print("stacked_attn:", stacked_attn.shape)
    
    # Average layers
    layer_avg = stacked_attn.mean(dim=0)  # [batch, heads, queries, patches]
    print("layer_avg:",layer_avg.shape)
    # Average heads
    head_avg = layer_avg.mean(dim=1)      # [batch, queries, patches]
    print("head_avg:",head_avg.shape)
    return head_avg.squeeze(0).cpu().numpy()  # [100, 400]

#  Coordinate Mapping 
class CoordinateMapper:
    def __init__(self, image_size=640, grid_size=20):
        self.patch_size = image_size // grid_size  # 32
        self.grid_size = grid_size
        
    def get_patch_rect(self, patch_id):
        """Convert patch index to image coordinates"""
        row = patch_id // self.grid_size
        col = patch_id % self.grid_size
        return (col * self.patch_size, 
                row * self.patch_size,
                self.patch_size,
                self.patch_size)

class BirdVisualizer:
    def __init__(self, class_names):
        self.cmap = plt.get_cmap('tab20')
        self.class_names = class_names
        self.mapper = CoordinateMapper()
        
    def denormalize(self, tensor):
        """Denormalize image tensor"""
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEVICE)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEVICE)
        return (tensor.squeeze().permute(1,2,0)*std + mean).clamp(0,1).cpu().numpy()
    
    def visualize(self, image_tensor, pred_labels, attn_maps):
        """Main visualization function"""
        # Prepare image data
        image = (self.denormalize(image_tensor) * 255).astype(np.uint8)
        
        # Create canvas
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(image)
        
        # Process each query
        for query_idx in range(100):
            label_idx = pred_labels[query_idx]
            if label_idx >= len(self.class_names):
                continue
                
            # Get main patch for current query
            main_patch = np.argmax(attn_maps[query_idx])
            x, y, w, h = self.mapper.get_patch_rect(main_patch)
            
            # Draw marker
            self._draw_patch(ax, x, y, w, h, label_idx)
            
        # Add legend
        self._add_legend(ax, np.unique(pred_labels))
        
        plt.title("Bird Part Patches Classification", fontsize=18)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def _draw_patch(self, ax, x, y, w, h, label_idx):
        """Draw single patch marker"""
        # Rectangle
        rect = mpatches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=self.cmap(label_idx/15),
            facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Text label
        ax.text(
            x + w/2, y + h/2,
            self.class_names[label_idx],
            color='white',
            fontsize=10,
            ha='center', va='center',
            bbox=dict(
                facecolor=self.cmap(label_idx/15),
                alpha=0.8,
                boxstyle='round,pad=0.3'
            )
        )
    
    def _add_legend(self, ax, detected_classes):
        """Generate dynamic legend"""
        legend_elements = [
            mpatches.Patch(
                color=self.cmap(i/15),
                label=f"{self.class_names[i]} (Class {i})"
            )
            for i in detected_classes if i < 15
        ]
        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.15, 1),
            loc='upper left',
            title="Detected Parts",
            fontsize=12,
            title_fontsize=14
        )

if __name__ == "__main__":
    # Initialize components
    model, cfg = load_san_model(CHECKPOINT_PATH)
    visualizer = BirdVisualizer([
        "back", "beak", "belly", "breast", "crown", "forehead",
        "left eye", "left leg", "left wing", "nape", "right eye",
        "right leg", "right wing", "tail", "throat"
    ])
    
    # Process input data
    inputs = preprocess_image(IMAGE_PATH)
    
    # Model inference
    with torch.no_grad():
        batched_inputs = [{"image": inputs.squeeze(0), "meta": {"dataset_name": "bird_parts"}}]
        output = model(batched_inputs)
    
    # Parse outputs
    mask_logits = output[0]["mask_logits"]
    attn_maps = process_attention_maps(output[0]["mask_attention_maps"])
    mask_logits = mask_logits.softmax(dim=1)
    pred_labels = mask_logits[:, :-1].argmax(dim=1).cpu().numpy()
    
    # Execute visualization
    print("Visualization Statistics:")
    print(f"- Unique classes detected: {len(np.unique(pred_labels))}")
    print(f"- Patch size: {visualizer.mapper.patch_size}x{visualizer.mapper.patch_size} pixels")
    
    visualizer.visualize(inputs, pred_labels, attn_maps)