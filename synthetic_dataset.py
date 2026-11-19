import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
CLASS_NAMES = ["PET", "HDPE", "LDPE", "PP", "PS", "Other Plastic", "Not Plastic Waste"]

CLASS_STYLES = {
    "PET": {"bg": (230, 245, 255), "item": (54, 124, 220), "accent": (22, 82, 160)},
    "HDPE": {"bg": (235, 255, 232), "item": (50, 180, 110), "accent": (31, 110, 74)},
    "LDPE": {"bg": (255, 248, 218), "item": (217, 170, 52), "accent": (139, 102, 10)},
    "PP": {"bg": (255, 238, 232), "item": (226, 120, 62), "accent": (163, 79, 28)},
    "PS": {"bg": (247, 236, 255), "item": (155, 110, 220), "accent": (86, 55, 150)},
    "Other Plastic": {"bg": (255, 228, 228), "item": (220, 95, 95), "accent": (142, 45, 45)},
    "Not Plastic Waste": {"bg": (245, 245, 245), "item": (120, 120, 120), "accent": (60, 60, 60)},
}


def draw_plastic_object(draw, width, height, base_color, accent_color, class_name):
    rng = np.random.default_rng()
    cx = rng.integers(50, width - 50)
    cy = rng.integers(50, height - 50)
    rx = rng.integers(35, 90)
    ry = rng.integers(25, 80)

    if class_name == "PET":
        draw.rounded_rectangle((cx - rx, cy - ry, cx + rx, cy + ry), radius=20, fill=base_color)
        draw.rounded_rectangle((cx - rx + 18, cy - ry + 18, cx + rx - 18, cy + ry - 18), radius=18, fill=accent_color)
        draw.line((cx - 20, cy - ry + 5, cx - 20, cy + ry - 5), fill=(255, 255, 255), width=6)
        draw.line((cx + 20, cy - ry + 5, cx + 20, cy + ry - 5), fill=(255, 255, 255), width=6)
    elif class_name == "HDPE":
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=base_color)
        for i in range(6):
            y = cy - ry + i * 24
            draw.line((cx - rx + 12, y, cx + rx - 12, y), fill=accent_color, width=3)
    elif class_name == "LDPE":
        draw.rounded_rectangle((cx - rx, cy - ry, cx + rx, cy + ry), radius=28, fill=base_color)
        for i in range(8):
            x = cx - rx + 10 + i * 18
            draw.line((x, cy - ry + 10, x, cy + ry - 10), fill=accent_color, width=3)
    elif class_name == "PP":
        draw.polygon([(cx - rx, cy), (cx, cy - ry), (cx + rx, cy), (cx, cy + ry)], fill=base_color)
        draw.line((cx, cy - ry, cx, cy + ry), fill=accent_color, width=4)
    elif class_name == "PS":
        for x in range(cx - rx, cx + rx + 1, 18):
            draw.rectangle((x, cy - ry, x + 10, cy + ry), fill=base_color)
        draw.rectangle((cx - rx, cy - ry, cx + rx, cy + ry), outline=accent_color, width=3)
    else:
        draw.rounded_rectangle((cx - rx, cy - ry, cx + rx, cy + ry), radius=18, fill=base_color)
        draw.ellipse((cx - rx // 2, cy - ry // 2, cx + rx // 2, cy + ry // 2), fill=accent_color)
        draw.line((cx - 35, cy + 10, cx + 35, cy + 10), fill=(255, 255, 255), width=5)


def generate_image(class_name, out_path, index):
    style = CLASS_STYLES[class_name]
    img = Image.new("RGB", (224, 224), style["bg"])
    draw = ImageDraw.Draw(img)

    for _ in range(10):
        x0 = np.random.randint(0, 220)
        y0 = np.random.randint(0, 220)
        x1 = x0 + np.random.randint(10, 80)
        y1 = y0 + np.random.randint(10, 80)
        draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255, 60), outline=None)

    draw_plastic_object(draw, 224, 224, style["item"], style["accent"], class_name)

    # add subtle blur / texture
    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    img = ImageOps.autocontrast(img)

    if index % 2 == 0:
        img = img.rotate(np.random.randint(-12, 13), expand=False)

    img.save(out_path)


def create_sample_dataset():
    for split in ["train", "val"]:
        for class_name in CLASS_NAMES:
            class_dir = (TRAIN_DIR if split == "train" else VAL_DIR) / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

    total = {"train": 180, "val": 40}
    for split, count in total.items():
        for class_name in CLASS_NAMES:
            target_dir = (TRAIN_DIR if split == "train" else VAL_DIR) / class_name
            for i in range(count):
                generate_image(class_name, target_dir / f"{class_name.lower()}_{i:03d}.png", i)

    print("Synthetic dataset created successfully.")
    print(f"Train images: {len(list(TRAIN_DIR.rglob('*.png')))}")
    print(f"Validation images: {len(list(VAL_DIR.rglob('*.png')))}")


if __name__ == "__main__":
    create_sample_dataset()
