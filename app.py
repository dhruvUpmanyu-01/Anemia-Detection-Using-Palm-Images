from flask import Flask, request, jsonify
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import mediapipe as mp
import base64
import os
from io import BytesIO

app = Flask(__name__)

# === Model path ===
model_path = "models/mobilenet_model.pth"
os.makedirs("cropped", exist_ok=True)

# === MediaPipe palm extraction ===
def extract_palm(image_bytes, save_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

    # Read image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        h, w, _ = image.shape
        palm_ids = [1, 5, 9, 13, 17]
        coords = [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_hand_landmarks[0].landmark]

        x_vals, y_vals = zip(*coords)
        x_min = max(min(x_vals) - 10, 0)
        x_max = min(max(x_vals) + 10, w)
        y_min = max(min(y_vals) - 10, 0)
        y_max = min(max(y_vals) + 10, h)

        palm_crop = image[y_min:y_max, x_min:x_max]
        cv2.imwrite(save_path, palm_crop)
        return palm_crop
    else:
        raise ValueError("No hand detected.")

# === Image transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Load model ===
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# === Prediction ===
def predict(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    input_tensor = transform(img_pil).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, dim=1)
        return pred.item(), confidence.item()

# === Convert image to base64 ===
def image_to_base64(img):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# === Endpoint ===
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        left_file = request.files.get("left")
        right_file = request.files.get("right")

        if not left_file or not right_file:
            return jsonify({"error": "Both left and right palm images are required"}), 400

        left_bytes = left_file.read()
        right_bytes = right_file.read()

        left_crop = extract_palm(left_bytes, "cropped/left_crop.png")
        right_crop = extract_palm(right_bytes, "cropped/right_crop.png")

        left_pred, left_conf = predict(left_crop)
        right_pred, right_conf = predict(right_crop)

        if left_pred == right_pred:
            final_pred = left_pred
            agg_method = "both_agree"
        else:
            final_pred = left_pred if left_conf > right_conf else right_pred
            agg_method = "confidence"

        label_map = {0: "anemic", 1: "non_anemic"}
        response = {
            "aggregation_method": agg_method,
            "confidence": round(max(left_conf, right_conf), 4),
            "prediction": label_map[final_pred],
            "cropped_base64": {
                "left": image_to_base64(left_crop),
                "right": image_to_base64(right_crop)
            }
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=2000)
