#!/root/HealthcareApp/.venv/bin/python

from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
from werkzeug.security import check_password_hash
from dbFunctions import init__users_db, init_patients_db, get_user_by_email, get_patients, get_patient_by_id
from models import Patient
from utility import bytes_to_base64, non_max_suppression, preprocess_image, create_saliency_map
import tensorflow as tf
import pydicom
from io import BytesIO
import base64
# Add YOLO and SAM imports
from ultralytics import YOLO
import torch
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


DATABASE = "./Backend/users.db"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)
# Autorizza solo il dominio del frontend
CORS(app, origins=["*"], methods=["GET", "POST", "PUT", "DELETE"])

def letterbox(img, new_shape=(416, 416), color=(0, 0, 0)):
    '''Resize image to fit in new_shape with unchanged aspect ratio using padding.'''
    shape = img.shape[:2]  # current shape [height, width]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

@app.route("/", methods=["GET"])
def hello():
    return jsonify(message="Hello, World from Flask API")

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    is_doctor = data.get("isDoctor")

    if not email or not password or is_doctor is None:
        print("Login failed: Missing credentials")
        return jsonify(success=False, message="Missing credentials"), 400

    user = get_user_by_email(email)
    if not user:
        print("Login failed: User not found")
        return jsonify(success=False, message="User not found"), 404
    elif user.is_doctor != is_doctor:
        print("Login failed: User type mismatch")
        print(f"Expected is_doctor: {is_doctor}, found: {user.is_doctor}")
        return jsonify(success=False, message="User type mismatch"), 403
    elif user and not user.is_doctor:
        print("Login failed: User is not authorized")
        return jsonify(success=False, message="User is not a doctor"), 403
    elif user and check_password_hash(user.password, password):
        global current_user
        current_user = {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "surname": user.surname,
            "is_doctor": user.is_doctor
        }
        return jsonify(success=True, message="Login successful")
    else:
        return jsonify(success=False, message="Invalid credentials"), 401
    
@app.route("/user", methods=["GET"])
def user():
    if current_user:
        return jsonify(success=True, message="Welcome to the dashboard", user=current_user)
    else:
        return jsonify(success=False, message="Unauthorized"), 401
    
@app.route("/patients", methods=["GET"])
def patients():
    patients = get_patients()
    if not patients:
        return jsonify(success=False, message="No patients found"), 404
    else:
        # Convert each patient to a dict, exclude 'image' field if present
        patients_list = []
        for p in patients:
            d = vars(p).copy()
            if 'image' in d:
                del d['image']
            patients_list.append(d)
        return jsonify(success=True, patients=patients_list)

@app.route("/patient/<patient_id>", methods=["GET"])
def patient(patient_id):
    patient = get_patient_by_id(patient_id)
    if not patient:
        return jsonify(success=False, message="Patient not found"), 404
    else:
        patient_dict = vars(patient).copy()
        patient_dict = bytes_to_base64(patient_dict)
        return jsonify(success=True, patient=patient_dict)
    
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get('file')
    model = request.form.get('model')
    if model not in ["vgg", "resnet", "densenet"]:
        return jsonify(success=False, message="Invalid model specified"), 400
    if not file:
        return jsonify(success=False, message="No image uploaded"), 400
    file_bytes = file.read()
    dicom_stream = BytesIO(file_bytes)
    try:
        patient = Patient.from_dicom(dicom_file=dicom_stream)
        print("DICOM file read, Patient id: ", patient.id)
        patient_dict = vars(patient).copy()
        # Preprocess the image for prediction
        dicom_stream.seek(0)
        ds = pydicom.dcmread(dicom_stream)
        print("DICOM file read")
        preprocessed_image = preprocess_image(ds.pixel_array)
        print("Image preprocessed")
        # --- YOLO detection ---
        # Use original DICOM pixel array for YOLO and overlay
        arr = ds.pixel_array
        # --- Ensure image is uint8, 0-255, 3-channel RGB for YOLO and SAM ---
        if arr.ndim == 2:
            arr_rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr_rgb = cv2.cvtColor(arr.squeeze(-1), cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[-1] >= 3:
            arr_rgb = arr[..., :3]
        else:
            arr_rgb = arr
        # Normalize only if needed, then convert to uint8
        if arr_rgb.dtype != np.uint8:
            arr_rgb = arr_rgb.astype(np.float32)
            arr_rgb = arr_rgb - arr_rgb.min()
            if arr_rgb.max() > 0:
                arr_rgb = arr_rgb / arr_rgb.max() * 255
            arr_rgb = arr_rgb.astype(np.uint8)
        # Letterbox resize to 416x416 for YOLO/SAM, keep scale and pad
        arr_rgb_416, ratio, (dw, dh) = letterbox(arr_rgb, new_shape=(416, 416))
        results = yolo_model.predict(source=arr_rgb_416, save=False, imgsz=416, conf=0.01, device=0)
        predicted_boxes = results[0].boxes
        # Use the first box (highest confidence)
        if len(predicted_boxes) == 0:
            return jsonify(success=False, message="No bounding box detected by YOLO"), 400
        nms_boxes = non_max_suppression(predicted_boxes, iou_threshold=0.2)
        bbox = nms_boxes[0]
        h, w = arr_rgb_416.shape[:2]  # 416, 416
        # If normalized, convert to pixel
        if max(bbox) <= 1.0:
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
        else:
            x1, y1, x2, y2 = map(int, bbox)
        # Pass the 416x416 image and bbox to SAM
        bbox_prompt = np.array([[x1, y1, x2, y2]])
        predictor.set_image(arr_rgb_416)
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox_prompt,
            multimask_output=False
        )
        # Get the mask (first and only mask since multimask_output=False)
        sam_mask_416 = masks[0]  # This is already in 416x416 space
        # --- Overlay bbox and mask on original image size ---
        # Use original DICOM image for overlay
        if arr.ndim == 2:
            orig_img = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            orig_img = cv2.cvtColor(arr.squeeze(-1), cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[-1] >= 3:
            orig_img = arr[..., :3]
        else:
            orig_img = arr
        orig_img = orig_img.astype(np.uint8)
        h_orig, w_orig = orig_img.shape[:2]
        # Map bbox from 416x416 letterbox space to original size (remove padding, then scale)
        # Remove padding
        x1_unpad = (x1 - dw)
        y1_unpad = (y1 - dh)
        x2_unpad = (x2 - dw)
        y2_unpad = (y2 - dh)
        # Scale back to original image size
        x1o = int(x1_unpad / ratio)
        y1o = int(y1_unpad / ratio)
        x2o = int(x2_unpad / ratio)
        y2o = int(y2_unpad / ratio)
        # --- Transform mask from 416x416 letterbox space to original image size ---
        # sam_mask_416 is boolean mask in 416x416 letterbox space
        mask_probs = sam_mask_416.astype(np.float32)
        
        # Debug prints
        print(f"SAM mask shape: {mask_probs.shape}")
        print(f"Original image shape: {arr_rgb.shape}")
        print(f"Letterbox ratio: {ratio}, padding: dw={dw}, dh={dh}")
        
        # Calculate the unpadded dimensions
        unpad_h = int(round(arr_rgb.shape[0] * ratio))
        unpad_w = int(round(arr_rgb.shape[1] * ratio))
        pad_left = int(round(dw))
        pad_top = int(round(dh))
        
        # Ensure valid cropping indices
        if pad_top + unpad_h > mask_probs.shape[0]:
            unpad_h = mask_probs.shape[0] - pad_top
        if pad_left + unpad_w > mask_probs.shape[1]:
            unpad_w = mask_probs.shape[1] - pad_left
            
        # First crop the mask to remove padding
        try:
            mask_unpadded = mask_probs[pad_top:pad_top+unpad_h, pad_left:pad_left+unpad_w]
            print(f"Unpadded mask shape: {mask_unpadded.shape}")
            
            # Ensure mask is not empty and has valid dimensions
            if mask_unpadded.size == 0 or w_orig <= 0 or h_orig <= 0:
                raise ValueError("Invalid mask dimensions")
                
            # Then resize to original image dimensions
            mask_probs_orig = cv2.resize(mask_unpadded, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            
        except Exception as e:
            print(f"Error processing mask: {str(e)}")
            # Fallback: use a simple threshold on the entire mask
            mask_probs_orig = cv2.resize(mask_probs, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        
        # Continue with overlay creation
        pred_mask = mask_probs_orig > 0.5
        overlay_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        alpha = 0.7
        red_mask = np.zeros_like(overlay_img)
        red_mask[..., 2] = 255  # Red in BGR
        mask_indices = pred_mask.astype(bool)
        overlay_img[mask_indices] = cv2.addWeighted(overlay_img[mask_indices], 1 - alpha, red_mask[mask_indices], alpha, 0)
        cv2.rectangle(overlay_img, (x1o, y1o), (x2o, y2o), (0, 255, 0), 2)
        # Convert back to RGB for PIL
        overlay_img_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
        overlay_img_pil = Image.fromarray(overlay_img_rgb)
        buf = BytesIO()
        overlay_img_pil.save(buf, format='PNG')
        overlay_img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        patient_dict["image_base64"] = overlay_img_base64
        # Convert any bytes fields to base64 for JSON serialization
        patient_dict = bytes_to_base64(patient_dict)
        # --- Classification and saliency map (unchanged) ---
        if model == "vgg":
            vgg_pred = vgg.predict(preprocessed_image)
            print("VGG prediction done")
        elif model == "resnet":
            vgg_pred = resnet.predict(preprocessed_image)
            print("ResNet prediction done")
        elif model == "densenet":
            vgg_pred = densenet.predict(preprocessed_image)
            print("DenseNet prediction done")
        if vgg_pred > 0.5:
            predicted_class = "Malignant"
            probability = float(vgg_pred[0][0])
        else:
            predicted_class = "Benign"
            probability = float(1 - vgg_pred[0][0])
        chosen_model = vgg if model == "vgg" else resnet if model == "resnet" else densenet
        saliency_map = create_saliency_map(chosen_model, preprocessed_image)
        patient_dict["saliency_map_base64"] = saliency_map
        # Save overlay image to disk for debugging
        overlay_img_pil.save("debug_overlay.png", format='PNG')
        return jsonify(success=True, predicted_class=predicted_class, probability=probability, patient=patient_dict)
    except Exception as e:
        return jsonify(success=False, message=f"Prediction failed: {str(e)}"), 500


if __name__ == "__main__":
    # Gunicorn in produzione, qui per sviluppo rapido
    init_patients_db()
    init__users_db()
    global vgg, resnet, densenet
    vgg = tf.keras.models.load_model("./Backend/Models/vgg16_fold5_best.keras")
    resnet = tf.keras.models.load_model("./Backend/Models/resnet_fine_tuned.keras")
    densenet = tf.keras.models.load_model("./Backend/Models/densenet_fine_tuned.keras")
    # Load YOLO and SAM models
    global yolo_model, predictor, sam_device
    yolo_model = YOLO("./Backend/Models/best_noaug.pt")  # Update with your YOLO weights path if needed
    sam_device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_config = "configs/sam2.1/sam2.1_hiera_b+.yaml"  # Update as needed
    sam2_ckpt = "./Backend/Models/sam2_finetuned_ultrasound_best.pt"     # Update as needed
    sam2_model = build_sam2(sam2_config, None, device=sam_device)
    state_dict = torch.load(sam2_ckpt, map_location="cpu")
    sam2_model.load_state_dict(state_dict)
    sam2_model = sam2_model.to(sam_device)
    predictor = SAM2ImagePredictor(sam2_model)
    app.run(host="0.0.0.0", port=5000)