import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import mediapipe as mp

# ==== PATHS ====
left_image_path = r"D:\Anaemia Detection\Palm Image Extractor\test_images\left.png"
right_image_path = r"D:\Anaemia Detection\Palm Image Extractor\test_images\right.png"
model_path = r"D:\Anaemia Detection\Palm Image Extractor\models\mobilenet_model.pth"
crop_dir = r"D:\Anaemia Detection\Palm Image Extractor\cropped"
os.makedirs(crop_dir, exist_ok=True)

# ==== PALM CROP FUNCTION ====
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

        # More centered palm region - tighter crop
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
        cv2.imwrite(output_path, palm_crop)
        return palm_crop
    else:
        raise ValueError(f"No hand detected in image: {image_path}")


# ==== LOAD MODEL ====
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# ==== TRANSFORM ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ==== PREDICT FUNCTION ====
def predict(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    input_tensor = transform(img_pil).unsqueeze(0)

    with torch.no_grad():
        out = model(input_tensor)
        probs = torch.softmax(out, dim=1)
        confidence, pred = torch.max(probs, dim=1)
        return pred.item(), confidence.item()

# ==== MAIN PIPELINE ====
try:
    print("🟡 Extracting left palm...")
    left_crop_path = os.path.join(crop_dir, "left_crop.png")
    left_crop = extract_palm(left_image_path, left_crop_path)
    print("✅ Left palm cropped and saved")

    print("🟡 Extracting right palm...")
    right_crop_path = os.path.join(crop_dir, "right_crop.png")
    right_crop = extract_palm(right_image_path, right_crop_path)
    print("✅ Right palm cropped and saved")

    print("🧠 Predicting on left palm...")
    left_pred, left_conf = predict(left_crop)
    print(f"   ➤ Left: {left_pred}, Confidence: {left_conf:.4f}")

    print("🧠 Predicting on right palm...")
    right_pred, right_conf = predict(right_crop)
    print(f"   ➤ Right: {right_pred}, Confidence: {right_conf:.4f}")

    # === AGGREGATION ===
    if left_pred == right_pred:
        final_pred = left_pred
    else:
        final_pred = 0 if left_conf > right_conf else 1  # Higher confidence wins

    label_map = {0: "anemic", 1: "non_anemic"}
    print(f"\n🩺 Final Diagnosis: {label_map[final_pred]}")

except Exception as e:
    print(f"❌ Pipeline failed: {e}")
