import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms
import open_clip

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load pre-trained model
model_path = "model/san_vit_b_16.pth"
assert os.path.exists(model_path), f"Model file not found at {model_path}"
print(f"Loading model from {model_path}")

model = open_clip.create_model('ViT-B-16', pretrained=False)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.to(device)
# set the model to evaluation mode
model.eval()

# preprocess image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # resize to 224x224 for ViT
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# load image and preprocess
image_path = "D:\\Github\\SAN-evaluation\\data\\images\\195.Carolina_Wren\\Carolina_Wren_0011_186871.jpg"
original_image = Image.open(image_path).convert("RGB")
inputs = preprocess(original_image).unsqueeze(0).to(device)

# text descriptions for patches
text_descriptions = [
    "back", "beak", "belly", "breast", "crown", "forehead",
    "left eye", "left leg", "left wing", "nape", "right eye",
    "right leg", "right wing", "tail", "throat"
]
# tokenize text descriptions
text_tokens = open_clip.tokenize(text_descriptions).to(device)

# encode text tokens to get text features
with torch.no_grad():
    text_features = model.encode_text(text_tokens)

# project text features to match visual features
input_dim = text_features.shape[-1]  
projection_layer = torch.nn.Linear(input_dim, 768).to(device)
text_features = projection_layer(text_features)

print(f"text_features.shape after projection: {text_features.shape}")  # [15, 768]

# modify the visual forward pass to get all patch features
def modified_visual_forward(self, x):
    x = self.conv1(x)  # [batch, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [batch, width, grid^2]
    x = x.permute(0, 2, 1)  # [batch, grid^2, width]
    x = torch.cat([
        self.class_embedding.to(x.dtype) + 
        torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
        x
    ], dim=1)  # [batch, grid^2+1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)
    
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    return x.permute(1, 0, 2)  # LND -> NLD

# replace the forward pass with the modified version
model.visual.forward = modified_visual_forward.__get__(model.visual, type(model.visual))

# get all patch features
with torch.no_grad():
    all_features = model.visual(inputs)  # [1, 197, 768] (CLS + 196 patches)
    patch_features = all_features[:, 1:]  # remove CLS token [1, 196, 768]

print(f"patch_features.shape: {patch_features.shape}")  # [1, 196, 768]

# calculate similarity between patch features and text features
similarity = (patch_features @ text_features.T).squeeze(0)  # [196, 15]
predicted_labels = similarity.argmax(dim=-1).cpu().numpy()

# denormalize image
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3).to(device)
    return (tensor.squeeze().permute(1, 2, 0) * std + mean).clamp(0, 1).cpu().numpy()

processed_image = denormalize(inputs)
resized_image = (processed_image * 255).astype(np.uint8)

# visualize patches with predicted labels
def visualize_patches(image_array, labels, patch_size=16):
    plt.figure(figsize=(15, 15))
    ax = plt.gca()
    ax.imshow(image_array)
    
    num_patches = int(np.sqrt(len(labels)))  # 14 for 224x224 with 16x16 patches
    actual_size = image_array.shape[0]  # Ensure processed image size is correct
    
    for i in range(num_patches):
        for j in range(num_patches):
            idx = i * num_patches + j
            label = text_descriptions[predicted_labels[idx]]
            
            # Compute coordinates (based on processed image)
            x = j * patch_size
            y = i * patch_size
            
            # Draw bounding box
            rect = mpatches.Rectangle(
                (x, y), patch_size, patch_size,
                linewidth=1, edgecolor='white', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(
                x + patch_size/2, y + patch_size/2, label,
                color='blue', fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
            )
    
    plt.axis('off')
    plt.show()

# Execute visualization
visualize_patches(resized_image, predicted_labels)