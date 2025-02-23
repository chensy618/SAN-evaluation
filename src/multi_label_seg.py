import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

# 导入 SAN 配置 & 模型
from san.config import add_san_config
from san.model.san import SAN

# 从 utils.py 导入已经定义的 get_labelset_from_dataset
from san.model.clip_utils.utils import get_labelset_from_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载 SAN 模型
def load_san_model(checkpoint_path):
    cfg = get_cfg()
    add_san_config(cfg)
    model = SAN(cfg)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        raise ValueError("Checkpoint file does not contain 'model' key.")

    model.to(DEVICE)
    model.eval()
    return model, cfg

# 2. 预处理图像
def preprocess_image(image_path, resize=(640, 640)):
    orig_image = Image.open(image_path).convert("RGB")
    W, H = orig_image.size
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(orig_image).unsqueeze(0).to(DEVICE)
    return img_tensor, (H, W), orig_image

# 3. 多通道后处理 (softmax + 阈值 + 调试信息)
def get_multi_label_masks(model, image_tensor, dataset_name="bird_parts", threshold=0.1):
    """
    1) 对输出做 softmax 转概率
    2) 逐通道二值化 (prob[c] > threshold)
    3) 仅在 bird_mask 内再看其他部位
    4) 打印每个通道的最大 logits/最大 prob, 便于调试
    5) 返回 mask 字典
    """
    batched_inputs = [{
        "image": image_tensor.squeeze(0),
        "meta": {"dataset_name": dataset_name}
    }]

    with torch.no_grad():
        outputs = model(batched_inputs)
        logits = outputs[0]["sem_seg"]  # shape [C, H, W]

    labelset = get_labelset_from_dataset(dataset_name)
    C = logits.shape[0]
    if len(labelset) != C:
        print(f"Warning: labelset has {len(labelset)} classes, but logits has {C} channels!")
        print("They must match. Otherwise the following index(...) calls may fail.")

    # ========== 调试：打印每个通道的最大logits、最大prob ==========
    # 先看每个通道的logits max
    for i in range(C):
        print(f"[DEBUG] {labelset[i]}: logits.max()={logits[i].max().item():.4f}, "
              f"logits.min()={logits[i].min().item():.4f}")

    # softmax 转成概率
    prob = F.softmax(logits, dim=0)

    # 再看每个通道的 prob max
    for i in range(C):
        print(f"[DEBUG] {labelset[i]}: prob.max()={prob[i].max().item():.4f}, "
              f"prob.min()={prob[i].min().item():.4f}")

    # 获取各标签的索引
    bird_idx       = labelset.index("bird")
    back_idx       = labelset.index("back")
    beak_idx       = labelset.index("beak")
    left_eye_idx   = labelset.index("left eye")
    right_eye_idx  = labelset.index("right eye")
    left_leg_idx   = labelset.index("left leg")
    right_leg_idx  = labelset.index("right leg")
    left_wing_idx  = labelset.index("left wing")
    right_wing_idx = labelset.index("right wing")
    nape_idx       = labelset.index("nape")
    throat_idx     = labelset.index("throat")
    belly_idx      = labelset.index("belly")
    breast_idx     = labelset.index("breast")
    crown_idx      = labelset.index("crown")
    forehead_idx   = labelset.index("forehead")
    tail_idx       = labelset.index("tail")
    bg_idx         = labelset.index("background")

    # Bird 区域
    bird_mask = (prob[bird_idx] > threshold)
    # Background
    background_mask = (prob[bg_idx] > threshold)

    # 在 bird 内再看其他部位
    back_mask      = (prob[back_idx]      > threshold) & bird_mask
    beak_mask      = (prob[beak_idx]      > threshold) & bird_mask
    left_eye_mask  = (prob[left_eye_idx]  > threshold) & bird_mask
    right_eye_mask = (prob[right_eye_idx] > threshold) & bird_mask
    left_leg_mask  = (prob[left_leg_idx]  > threshold) & bird_mask
    right_leg_mask = (prob[right_leg_idx] > threshold) & bird_mask
    left_wing_mask = (prob[left_wing_idx] > threshold) & bird_mask
    right_wing_mask= (prob[right_wing_idx]> threshold) & bird_mask
    nape_mask      = (prob[nape_idx]      > threshold) & bird_mask
    throat_mask    = (prob[throat_idx]    > threshold) & bird_mask
    belly_mask     = (prob[belly_idx]     > threshold) & bird_mask
    breast_mask    = (prob[breast_idx]    > threshold) & bird_mask
    crown_mask     = (prob[crown_idx]     > threshold) & bird_mask
    forehead_mask  = (prob[forehead_idx]  > threshold) & bird_mask
    tail_mask      = (prob[tail_idx]      > threshold) & bird_mask

    # 收集
    masks = {
        "bird": bird_mask.cpu().numpy(),
        "background": background_mask.cpu().numpy(),
        "back": back_mask.cpu().numpy(),
        "beak": beak_mask.cpu().numpy(),
        "left eye": left_eye_mask.cpu().numpy(),
        "right eye": right_eye_mask.cpu().numpy(),
        "left leg": left_leg_mask.cpu().numpy(),
        "right leg": right_leg_mask.cpu().numpy(),
        "left wing": left_wing_mask.cpu().numpy(),
        "right wing": right_wing_mask.cpu().numpy(),
        "nape": nape_mask.cpu().numpy(),
        "throat": throat_mask.cpu().numpy(),
        "belly": belly_mask.cpu().numpy(),
        "breast": breast_mask.cpu().numpy(),
        "crown": crown_mask.cpu().numpy(),
        "forehead": forehead_mask.cpu().numpy(),
        "tail": tail_mask.cpu().numpy(),
    }

    # ========== 调试：打印每个mask的像素数 ==========
    for lname, m in masks.items():
        print(f"[DEBUG] mask[{lname}].sum() = {m.sum()}")

    return masks

# 4. 可视化（为每个部位定义显眼颜色，统一 alpha=0.5）
def visualize_multi_masks(image_path, masks, orig_size):
    orig_image = Image.open(image_path).convert("RGB")
    w, h = orig_size[1], orig_size[0]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(orig_image)

    # 为每个部位定义专门颜色（尽量区分明显）
    color_map = {
        "bird":       (1.0, 0.0, 0.0),   # 红
        "background": (1.0, 1.0, 0.0),   # 黄
        "back":       (0.0, 0.0, 1.0),   # 蓝
        "beak":       (0.0, 1.0, 0.0),   # 绿
        "left eye":   (1.0, 0.0, 1.0),   # 品红
        "right eye":  (0.0, 1.0, 1.0),   # 青
        "left leg":   (0.8, 0.4, 0.0),
        "right leg":  (0.6, 0.6, 0.0),
        "left wing":  (0.6, 0.0, 0.6),
        "right wing": (0.2, 0.4, 0.8),
        "nape":       (0.4, 0.4, 0.4),
        "throat":     (0.9, 0.6, 0.0),
        "belly":      (0.1, 0.7, 0.5),
        "breast":     (0.8, 0.0, 0.4),
        "crown":      (0.3, 0.8, 0.3),
        "forehead":   (0.3, 0.3, 0.8),
        "tail":       (0.7, 0.3, 0.6),
    }
    alpha = 0.5

    for label_name, mask_arr in masks.items():
        mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8))
        mask_resized = mask_pil.resize((w, h), resample=Image.NEAREST)
        mask_bool = np.array(mask_resized) > 127

        # 如果想只画有像素的部位，可以做个判断
        if mask_bool.sum() == 0:
            continue

        r, g, b = color_map.get(label_name, (1.0, 1.0, 1.0))  # 找不到就白色
        overlay = np.zeros((h, w, 4), dtype=np.float32)
        overlay[..., 0] = r
        overlay[..., 1] = g
        overlay[..., 2] = b
        overlay[..., 3] = mask_bool.astype(float) * alpha

        ax.imshow(overlay)

    ax.set_title("Multi-Label Bird Parts Segmentation (Debug Mode)")
    ax.axis("off")
    plt.show()

# 5. 主函数
if __name__ == "__main__":
    CHECKPOINT_PATH = "model/san_vit_b_16.pth"
    IMAGE_PATH = "D:\\Github\\SAN-evaluation\\data\\images\\195.Carolina_Wren\\Carolina_Wren_0011_186871.jpg"

    # 加载模型
    san_model, _ = load_san_model(CHECKPOINT_PATH)

    # 读图 & 预处理
    img_tensor, orig_size, _ = preprocess_image(IMAGE_PATH)

    # 得到多通道掩码 (阈值改成 0.1; 你可尝试更低 0.05, 更高 0.2 等)
    masks = get_multi_label_masks(san_model, img_tensor, dataset_name="bird_parts", threshold=0.1)

    # 可视化
    visualize_multi_masks(IMAGE_PATH, masks, orig_size)
