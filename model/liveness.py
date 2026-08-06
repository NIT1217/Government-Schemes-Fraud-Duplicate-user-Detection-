import os
import sys
import cv2
import numpy as np

# Add Project Root

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Project Root:", project_root)


# Import Silent Face Anti-Spoofing Modules

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name


# Load Model (Only Once)

MODEL_DIR = os.path.join(project_root, "resources", "anti_spoof_models")

print("Loading Liveness Model...")

os.chdir(project_root)

model = AntiSpoofPredict(device_id=0)

image_cropper = CropImage()

print("Liveness Model Loaded Successfully")



# Liveness Detection Function

def check_liveness(image_path):

    # Read Image
    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Unable to read image.")

    # Detect Face
    image_bbox = model.get_bbox(image)

    prediction = np.zeros((1, 3))

    # Predict using all anti-spoofing models
    for model_name in os.listdir(MODEL_DIR):

        model_path = os.path.join(MODEL_DIR, model_name)

        h_input, w_input, model_type, scale = parse_model_name(model_name)

        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }

        img = image_cropper.crop(**param)

        prediction += model.predict(img, model_path)

    # Final Prediction
    label = np.argmax(prediction)

    score = float(prediction[0][label])

    # Verification
    if label == 1:
        liveness_verified = True
    else:
        liveness_verified = False

    return score, liveness_verified


