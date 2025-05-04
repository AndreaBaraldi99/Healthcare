import numpy as np
from PIL import Image
import io
import base64
import tensorflow as tf
import matplotlib.pyplot as plt

def bytes_to_base64(patient_dict):
    """
    Convert bytes to base64 string. Resize to 256x256 before encoding.
    """
    if 'image' in patient_dict and patient_dict['image']:
        try:
            # Determine dtype
            expected_len_16 = patient_dict['image_height'] * patient_dict['image_width'] * 2
            dtype = np.uint16 if len(patient_dict['image']) == expected_len_16 else np.uint8
            arr = np.frombuffer(patient_dict['image'], dtype=dtype)
            arr = arr.reshape((patient_dict['image_height'], patient_dict['image_width']))
            # Normalize to 0-255 for display
            arr = arr.astype(np.float32)
            arr = arr - arr.min()
            if arr.max() > 0:
                arr = arr / arr.max() * 255
            arr = arr.astype(np.uint8)
            img = Image.fromarray(arr, mode='L')
            img = img.resize((256, 256), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            patient_dict['image_base64'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            # Update image size in patient_dict
            patient_dict['image_height'] = 256
            patient_dict['image_width'] = 256
        except Exception as e:
            print(f'Image conversion error: {e}')
            patient_dict['image_base64'] = ""
    if 'image' in patient_dict:
        del patient_dict['image']
    return patient_dict

def preprocess_image(arr):
    """
    Preprocess a NumPy array image for prediction: resize to 256x256 and convert to a float32 tensor, no normalization.
    """
    # Ensure arr is a NumPy array
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    # Convert to PIL Image and resize
    img = Image.fromarray(arr)
    img = img.resize((256, 256), Image.LANCZOS)
    arr = np.array(img)
    # Add channel dimension if needed
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    # Add batch dimension
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return tf.convert_to_tensor(arr, dtype=tf.float32)

def create_saliency_map(model, image_tensor):
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
    grads = tape.gradient(prediction, image_tensor)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0] # shape: (256, 256)
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency))
    saliency_map_np = saliency.numpy()
    saliency_map_np = np.ma.masked_where(saliency_map_np < 0.1, saliency_map_np)
    # Take figsize from the original image
    orig = image_tensor[0].numpy()
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
    ax.imshow(orig / 255.0)  # Normalize the image for display
    ax.imshow(saliency_map_np, cmap='hot', vmax = 0.7)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    saliency_map_base64 = base64.b64encode(buf.read()).decode('utf-8')
    # Plot and overlay using matplotlib, save to buffer
    return saliency_map_base64

