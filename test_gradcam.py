import os
import numpy as np
from PIL import Image
import tensorflow as tf

def test_cam():
    model = tf.keras.models.load_model("models/brain_tumor_model.keras")
    base_model = model.layers[1]
    last_conv_layer = base_model.get_layer("out_relu")
    
    # 1. Feature extractor submodel
    conv_submodel = tf.keras.Model(inputs=base_model.input, outputs=last_conv_layer.output)

    # 2. Classification head submodel
    head_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = model.get_layer("global_avg_pool")(head_input)
    x = model.get_layer("dropout_1")(x)
    x = model.get_layer("dense_64")(x)
    head_output = model.get_layer("predictions")(x)
    head_model = tf.keras.Model(inputs=head_input, outputs=head_output)

    # Test with glioma sample
    img_path = "static/samples/glioma_sample.jpg"
    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB").resize((224, 224))
        arr = np.array(img_rgb, dtype=np.float32)
        arr = (arr / 127.5) - 1.0
        input_tensor = np.expand_dims(arr, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs = conv_submodel(input_tensor)
        tape.watch(conv_outputs)
        preds = head_model(conv_outputs)
        top_idx = tf.argmax(preds[0])
        score = preds[:, top_idx]

    grads = tape.gradient(score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
    heatmap = np.maximum(heatmap.numpy(), 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    
    print("[+] Grad-CAM shape:", heatmap.shape, "min:", heatmap.min(), "max:", heatmap.max())
    
    # Find bounding box / tumor localization
    # Threshold heatmap at 50% max activation
    binary_mask = (heatmap > 0.45).astype(np.uint8)
    # Resize mask to (224, 224)
    mask_img = Image.fromarray(binary_mask * 255).resize((224, 224), resample=Image.Resampling.BILINEAR)
    mask_arr = np.array(mask_img) > 128
    
    y_indices, x_indices = np.where(mask_arr)
    if len(x_indices) > 0 and len(y_indices) > 0:
        xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))
        ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        print(f"[+] Detected Tumor Box: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}], center=({center_x}, {center_y})")
    else:
        print("[-] No focal region detected")

if __name__ == "__main__":
    test_cam()
