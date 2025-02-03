import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # ✅ 显式导入，避免 `patches` 变量冲突
from PIL import Image
from torchvision import transforms
import open_clip

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load the model 
model_path = "model/san_vit_b_16.pth"
# model_path = "model/san_vit_large_14.pth"
assert os.path.exists(model_path), f"Model file not found at {model_path}"
print(f"Loading model from {model_path}")
checkpoint = torch.load(model_path, map_location=device, weights_only=True)

model = open_clip.create_model('ViT-B-16', pretrained=False)
# model = open_clip.create_model('ViT-L-14', pretrained=False)
model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.eval()

# load the image
# image_path = "D:\\Github\\SAN-evaluation\\data\\images\\001.Black_footed_Albatross\\Black_Footed_Albatross_0001_796111.jpg"
image_path = "D:\\Github\\SAN-evaluation\\data\\images\\004.Groove_billed_Ani\\Groove_Billed_Ani_0002_1670.jpg"
# image_path = "D:\\Github\\SAN-evaluation\\data\\images\\200.Common_Yellowthroat\\Common_Yellowthroat_0003_190521.jpg"
# image_path = "D:\\Github\\SAN-evaluation\\data\\images\\193.Bewick_Wren\\Bewick_Wren_0010_185142.jpg"
# image_path = "D:\\Github\\SAN-evaluation\\data\\images\\195.Carolina_Wren\\Carolina_Wren_0011_186871.jpg"
# image_path = "D:\\Github\\SAN-evaluation\\data\\images\\185.Bohemian_Waxwing\\Bohemian_Waxwing_0001_796680.jpg"
# image_path = "D:\\Github\\SAN-evaluation\\data\\images\\185.Bohemian_Waxwing\\Bohemian_Waxwing_0002_177986.jpg"
image = Image.open(image_path).convert("RGB")

# image processing
patch_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def compute_patch_size(image, num_patches=14):
    width, height = image.size
    return min(width // num_patches, height // num_patches)

# compute patch size
patch_size = compute_patch_size(image, num_patches=14)
# patch_size = compute_patch_size(image, num_patches=16)

def extract_patches(image, patch_size):
    # extract the patches
    image_tensor = patch_transform(image).unsqueeze(0).to(device)
    
    num_patches_x = image.width // patch_size
    num_patches_y = image.height // patch_size

    # align th image to the patch size
    unfold_x = num_patches_x * patch_size
    unfold_y = num_patches_y * patch_size
    image_tensor = image_tensor[:, :, :unfold_y, :unfold_x]
    
    patches = image_tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(1, 3, num_patches_x * num_patches_y, patch_size, patch_size)
    
    return patches.squeeze(0), num_patches_x, num_patches_y, image_tensor

# xtract patches
patches, num_patches_x, num_patches_y, inputs = extract_patches(image, patch_size)

# denormalize the image
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3).to(device)

denormalized_image = (inputs.squeeze(0).permute(1, 2, 0) * std + mean).clamp(0, 1).cpu().numpy()
denormalized_image = (denormalized_image * 255).astype(np.uint8)  # make sure it is the numpy array

# text descriptions
text_descriptions = [
    "back", "beak", "belly", "breast", "crown", "forehead",
    "left eye", "left leg", "left wing", "nape", "right eye",
    "right leg", "right wing", "tail", "throat"
]

text_tokens = open_clip.tokenize(text_descriptions).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens).to(device)

def get_patch_features(patches):
    features = []
    with torch.no_grad():
        for i in range(patches.shape[1]):
            patch = patches[:, i].cpu().numpy().transpose(1, 2, 0)  
            patch = Image.fromarray((patch * 255).astype('uint8'))  
            
            patch_tensor = patch_transform(patch).unsqueeze(0).to(device)  
            patch_tensor = torch.nn.functional.interpolate(patch_tensor, size=(224, 224), mode='bilinear')  
            
            output = model.encode_image(patch_tensor)  
            features.append(output.cpu().numpy().squeeze())  
    return np.array(features)

# compute patch features
patch_features = torch.tensor(get_patch_features(patches), device=device)

def classify_patches(patch_features):
    # similarity between patches and text descriptions
    with torch.no_grad():
        similarity = patch_features @ text_features.T
        predicted_labels = similarity.argmax(dim=-1)
        return [text_descriptions[i.item()] for i in predicted_labels]

# classify patches
labels = classify_patches(patch_features)

def calculate_patch_coordinates(image, patch_size):
    # compute the patch coordinates
    _, H, W = image.shape
    n_rows = H // patch_size
    n_cols = W // patch_size
    patch_coords = [(j * patch_size, i * patch_size) for i in range(n_rows) for j in range(n_cols)]
    return patch_coords

def visualize_patches_with_coordinates(image, patch_coords, labels, patch_size):
    fig, ax = plt.subplots(figsize=(15, 15))  
    ax.imshow(image)
    ax.set_title("Patch Labels Visualization", fontsize=16)

    # draw patches and labels
    for (x, y), label in zip(patch_coords, labels):
        # draw the grid
        rect = mpatches.Rectangle((x, y), patch_size, patch_size, linewidth=1, edgecolor='white', facecolor='none')
        ax.add_patch(rect)
        
        # draw the label
        ax.text(x + patch_size // 2, y + patch_size // 2, label,
                color="blue", fontsize=8, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.6))
        
        # ax.text(x + patch_size // 2, y + patch_size // 2, label,
        #         color="blue", fontsize=8, ha="center", va="center")

    plt.axis("off")
    plt.show()

# visualize the patches
patch_coords = calculate_patch_coordinates(inputs.squeeze(0), patch_size)
visualize_patches_with_coordinates(denormalized_image, patch_coords, labels, patch_size)

