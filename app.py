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
# === MediaPipe palm extraction with background removal ===
def extract_palm(image_path, output_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        h, w, _ = image.shape

        landmark_ids = [0, 1, 5, 9, 13, 17]
        hand_landmarks = results.multi_hand_landmarks[0]
        coords = [(int(hand_landmarks.landmark[i].x * w),
                   int(hand_landmarks.landmark[i].y * h)) for i in landmark_ids]

        x_vals, y_vals = zip(*coords)
        x_min = max(min(x_vals) - 20, 0)
        x_max = min(max(x_vals) + 20, w)
        y_min = max(min(y_vals) - 20, 0)
        y_max = min(max(y_vals) + 20, h)

        palm_crop = image[y_min:y_max, x_min:x_max]

        # === Background removal using HSV skin mask ===
        hsv = cv2.cvtColor(palm_crop, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Extract only palm
        palm_only = cv2.bitwise_and(palm_crop, palm_crop, mask=mask)

        # Replace background with white
        white_bg = np.ones_like(palm_crop, dtype=np.uint8) * 255
        mask_inv = cv2.bitwise_not(mask)
        bg_white = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)
        final_img = cv2.add(palm_only, bg_white)

        cv2.imwrite(output_path, final_img)
        return final_img
    else:
        raise ValueError(f"No hand detected in image: {image_path}")

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

# === /predict endpoint for SINGLE image ===
@app.route('/predict', methods=['POST'])
def predict_single_image():
    try:
        uploaded_file = request.files.get("palm")

        if not uploaded_file:
            return jsonify({"error": "Palm image is required"}), 400

        file_bytes = uploaded_file.read()
        temp_path = "temp_palm.png"
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        # Palm extraction
        crop = extract_palm(temp_path, "cropped/palm_crop.png")

        # Prediction
        pred, conf = predict(crop)

        label_map = {0: "anemic", 1: "non_anemic"}
        response = {
            "confidence": round(conf, 4),
            "prediction": label_map[pred],
            "cropped_base64": image_to_base64(crop)
        }

        os.remove(temp_path)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Start Flask ===
if __name__ == '__main__':
    app.run(debug=True, port=2000)
