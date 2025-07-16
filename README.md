# Anemia-Detection-Using-Palm-Images

This project detects whether a person is **anemic** or **non-anemic** using palm images of both hands. It uses **MediaPipe** for palm region extraction and a custom-trained **MobileNetV2** model for classification.

---

###########################

##  Project Structure

your_team_folder/
├── anemia_pipeline.py # VS Code script for direct testing
├── app.py # Flask API (POST endpoint for predictions)
├── test_images/ # Contains left.png and right.png input palms
│ ├── left.png
│ └── right.png
├── cropped/ # Saved cropped palm images and base64 .txt
├── models/ # Trained models
│ └── mobilenet_model.pth
├── requirements.txt # Python dependencies
└── README.md # This documentation file

##########################

---

## 🛠 How to Run

###  Run from Terminal

```bash
python anemia_pipeline.py


######################

 What it does:

Loads both palm images (left.png and right.png)

Extracts palm regions using MediaPipe landmarks

Runs classification model on both palms

Aggregates prediction based on confidence

Outputs final anemia status

#############################

Run Flask API
bash
Copy
Edit
python app.py
 Endpoint: http://127.0.0.1:2000/predict
Method: POST

 Send two files in Postman:

left: left palm image

right: right palm image

################################

API Response (JSON)
json
Copy
Edit
{
  "prediction": "non_anemic",
  "confidence": 0.91,
  "aggregation_method": "confidence_majority",
  "cropped_base64": {
    "left": "<base64 string>",
    "right": "<base64 string>"
  }
}
 It also saves:

Cropped images (left_crop.png, right_crop.png)

Their base64 in .txt files (for training reuse)

##################################

 Model Info
Model: MobileNetV2

Framework: PyTorch

Input: Cropped palm region (224x224 RGB)

Classes:

0 = anemic

1 = non_anemic

########################

 Setup
Install all required packages:

bash
Copy
Edit
pip install -r requirements.txt
(You must be in your virtual environment)

 If you face issues installing packages:
Create a virtual environment with Python 11:

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate        # On Windows
# OR
source venv/bin/activate     # On Linux/macOS

python --version             # Confirm it shows Python 11
pip install -r requirements.txt
