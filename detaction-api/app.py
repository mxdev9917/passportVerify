from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract
import os
from datetime import datetime

# Configuration
pytesseract.pytesseract.tesseract_cmd = "tesseract"
os.environ["TESSDATA_PREFIX"] = "./tessdata"
model = YOLO("best.pt")
REQUIRED_CLASSES = ["Passport", "Photo", "MRZ"]
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)

def parse_mrz(mrz_text: str):
    lines = [line.strip() for line in mrz_text.splitlines() if line.strip()]
    
    if len(lines) == 2 and all(len(line) >= 44 for line in lines):
        line1, line2 = lines[:2]
        return {
            "document_type": line1[0],
            "issuing_country": line1[2:5],
            "surname": line1[5:].split("<<")[0].replace("<", " ").strip(),
            "given_names": " ".join(line1[5:].split("<<")[1].split("<")).strip(),
            "passport_number": line2[0:9].replace("<", ""),
            "passport_number_valid": line2[9] == "<",
            "nationality": line2[10:13],
            "birth_date": f"{line2[13:15]}-{line2[15:17]}-{line2[17:19]}",
            "birth_date_valid": line2[19] == "<",
            "sex": line2[20],
            "expiry_date": f"{line2[21:23]}-{line2[23:25]}-{line2[25:27]}",
            "expiry_date_valid": line2[27] == "<",
            "personal_number": line2[28:42].replace("<", ""),
            "raw": mrz_text.strip()
        }
    return {
        "raw": mrz_text.strip(),
        "error": "Unable to parse MRZ format"
    }

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or "personal_image" not in request.files:
        return jsonify({
            "error": "Both passport and personal images are required",
            "message": "Verification failed",
            "verify": False,
            "photo_match": False
        }), 400

    try:
        # Process passport image
        passport_file = request.files["image"]
        passport_np = np.frombuffer(passport_file.read(), np.uint8)
        passport_img = cv2.imdecode(passport_np, cv2.IMREAD_COLOR)
        
        # Process personal image
        personal_file = request.files["personal_image"]
        personal_np = np.frombuffer(personal_file.read(), np.uint8)
        personal_img = cv2.imdecode(personal_np, cv2.IMREAD_COLOR)

        # Detect objects in passport image
        results = model(passport_img)[0]
        detections = {}
        class_names = model.names

        for box in results.boxes:
            cls_id = int(box.cls)
            class_name = class_names[cls_id]
            if class_name in REQUIRED_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections[class_name] = passport_img[y1:y2, x1:x2]

        response = {
            "message": "Verification successful",
            "debug": {},
            "mrz": {},
            "verify": True,
            "photo_match": True  # This would be set by your face matching logic
        }

        if "MRZ" in detections:
            mrz_img = detections["MRZ"]
            gray = cv2.cvtColor(mrz_img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding for better OCR
            _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            mrz_text = pytesseract.image_to_string(
                threshold,
                lang="mrz",
                config="--psm 6"
            )
            
            # Save debug image
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            mrz_filename = f"mrz_{ts}.jpg"
            mrz_path = os.path.join(OUTPUT_DIR, mrz_filename)
            cv2.imwrite(mrz_path, mrz_img)

            parsed_mrz = parse_mrz(mrz_text)
            response["mrz"] = parsed_mrz
            response["debug"] = {
                "mrz_text": mrz_text,
                "mrz_image": f"/output/{mrz_filename}"
            }

            # Here you would add your face matching logic between:
            # detections["Photo"] (from passport) and personal_img
            # Set response["photo_match"] accordingly

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({
            "error": str(e),
            "message": "Verification failed",
            "verify": False,
            "photo_match": False
        }), 500

@app.route('/output/<filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)