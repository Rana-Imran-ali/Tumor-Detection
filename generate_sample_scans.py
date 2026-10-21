"""
Generate realistic synthetic brain MRI test images for immediate demonstration.
"""
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_base_brain_mri():
    """Generates a base skull & brain parenchyma oval background."""
    img = Image.new("L", (256, 256), color=10)
    draw = ImageDraw.Draw(img)
    
    # Outer skull bone (bright ring)
    draw.ellipse((28, 20, 228, 236), fill=35, outline=160, width=4)
    # Inner diploic space / dura
    draw.ellipse((35, 27, 221, 229), fill=20, outline=60, width=3)
    # Brain parenchyma (gray matter / white matter baseline)
    draw.ellipse((42, 34, 214, 222), fill=75, outline=50)
    
    # Ventricular system (CSF is dark on T1, bright on T2 - let's make typical T1/FLAIR ventricles)
    # Left lateral ventricle
    draw.polygon([(122, 90), (115, 110), (110, 140), (122, 130)], fill=25)
    # Right lateral ventricle
    draw.polygon([(134, 90), (141, 110), (146, 140), (134, 130)], fill=25)
    # Midline fissure
    draw.line((128, 38, 128, 218), fill=40, width=2)
    
    # Add subtle organic brain sulci textures
    for angle in range(0, 360, 15):
        rad = math.radians(angle)
        x1 = int(128 + 70 * math.cos(rad))
        y1 = int(128 + 80 * math.sin(rad))
        x2 = int(128 + 50 * math.cos(rad))
        y2 = int(128 + 60 * math.sin(rad))
        draw.line((x1, y1, x2, y2), fill=50, width=2)
        
    return img

def create_glioma_sample(save_path):
    img = create_base_brain_mri()
    draw = ImageDraw.Draw(img)
    # Heterogeneous infiltrative mass in the right frontal/parietal lobe with surrounding edema
    # Edema halo
    draw.ellipse((135, 75, 195, 135), fill=110)
    # Central necrotic/hyperintense core
    draw.ellipse((145, 85, 185, 125), fill=190, outline=235, width=3)
    # Irregular tumor rim
    draw.polygon([(150, 90), (175, 85), (180, 115), (160, 120), (145, 105)], fill=220)
    
    img = img.filter(ImageFilter.GaussianBlur(radius=1.8))
    # Convert to RGB and add subtle medical scan noise
    arr = np.array(img, dtype=float)
    noise = np.random.normal(0, 5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr).convert("RGB").save(save_path, quality=95)

def create_meningioma_sample(save_path):
    img = create_base_brain_mri()
    draw = ImageDraw.Draw(img)
    # Extra-axial dural-based round hyperintense mass with dural tail along the left parietal calvarium
    # Dural tail enhancement along inner skull
    draw.arc((36, 60, 90, 170), start=90, end=270, fill=220, width=4)
    # Dense, well-circumscribed homogeneous contrast-enhancing mass
    draw.ellipse((42, 95, 96, 145), fill=225, outline=250, width=2)
    
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    arr = np.array(img, dtype=float)
    noise = np.random.normal(0, 4, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr).convert("RGB").save(save_path, quality=95)

def create_pituitary_sample(save_path):
    img = create_base_brain_mri()
    draw = ImageDraw.Draw(img)
    # Central skull base sellar / suprasellar mass expanding sella turcica
    draw.ellipse((112, 140, 144, 175), fill=215, outline=240, width=3)
    # Optic chiasm compression region
    draw.ellipse((118, 132, 138, 145), fill=160)
    
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    arr = np.array(img, dtype=float)
    noise = np.random.normal(0, 4, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr).convert("RGB").save(save_path, quality=95)

def create_healthy_sample(save_path):
    img = create_base_brain_mri()
    img = img.filter(ImageFilter.GaussianBlur(radius=1.4))
    arr = np.array(img, dtype=float)
    noise = np.random.normal(0, 4, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr).convert("RGB").save(save_path, quality=95)

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "static", "samples")
    os.makedirs(out_dir, exist_ok=True)
    create_glioma_sample(os.path.join(out_dir, "glioma_sample.jpg"))
    create_meningioma_sample(os.path.join(out_dir, "meningioma_sample.jpg"))
    create_pituitary_sample(os.path.join(out_dir, "pituitary_sample.jpg"))
    create_healthy_sample(os.path.join(out_dir, "healthy_normal_sample.jpg"))
    print("Sample MRI images created successfully in static/samples/")
