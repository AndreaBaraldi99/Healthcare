#!/root/HealthcareApp/.venv/bin/python

from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
from werkzeug.security import check_password_hash
from dbFunctions import init__users_db, init_patients_db, get_user_by_email, get_patients, get_patient_by_id
from models import Patient
from utility import bytes_to_base64, preprocess_image, create_saliency_map
import tensorflow as tf
import pydicom
from io import BytesIO
import base64


DATABASE = "./Backend/users.db"

app = Flask(__name__)
# Autorizza solo il dominio del frontend
CORS(app, origins=["*"], methods=["GET", "POST", "PUT", "DELETE"])

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
        patient_dict = bytes_to_base64(patient_dict)
        # Preprocess the image for prediction
        dicom_stream.seek(0)
        ds = pydicom.dcmread(dicom_stream)
        print("DICOM file read")
        preprocessed_image = preprocess_image(ds.pixel_array)
        print("Image preprocessed")
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
        return jsonify(success=True, predicted_class=predicted_class, probability=probability, patient=patient_dict)
    except Exception as e:
        return jsonify(success=False, message=f"Prediction failed: {str(e)}"), 500


if __name__ == "__main__":
    # Gunicorn in produzione, qui per sviluppo rapido
    init_patients_db()
    init__users_db()
    global vgg, resnet, densenet
    vgg = tf.keras.models.load_model("./Backend/Models/vgg16_fine_tuned.keras")
    resnet = tf.keras.models.load_model("./Backend/Models/resnet_fine_tuned.keras")
    densenet = tf.keras.models.load_model("./Backend/Models/densenet_fine_tuned.keras")
    app.run(host="0.0.0.0", port=5000)