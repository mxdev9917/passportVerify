from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract
import os
from datetime import datetime
import re

# Configuration
pytesseract.pytesseract.tesseract_cmd = "tesseract"
os.environ["TESSDATA_PREFIX"] = "./tessdata"
model = YOLO("best.pt")
REQUIRED_CLASSES = ["Passport", "Photo", "MRZ"]
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)

def parse_mrz(mrz_text: str):
    """Enhanced MRZ parser with comprehensive passport type detection"""
    lines = [line.strip() for line in mrz_text.splitlines() if line.strip()]
    
    result = {
        "passport_type": "Unknown",
        "birth_date": "",
        "expiry_date": "",
        "given_names": "",
        "issuing_country": "",
        "nationality": "",
        "document_number": "",
        "raw": mrz_text.strip(),
        "sex": "",
        "surname": ""
    }

    if not lines:
        return result

    # Comprehensive passport type mapping
    PASSPORT_TYPES = {
        # Single-character codes
        'P': 'Regular Passport',
        'V': 'Visa',
        'I': 'Identity Card',
        'A': 'Alien ID',
        'C': 'Permit',
        # Two-character codes
        'PO': 'Official Passport',
        'PD': 'Diplomatic Passport',
        'PT': 'Travel Document',
        'PL': 'Laotian Passport',
        'PS': 'Service Passport',
        'PV': 'Visa',
        'PA': 'Alien Passport',
        # Country-specific codes
        'PM': 'Military Passport',
        'PE': 'Emergency Passport'
    }

    GENDER_MAP = {
        'M': 'Male',
        'F': 'Female',
        '<': 'Unspecified',
        'X': 'Other'
    }

    def parse_date(date_str):
        """Robust date parser with validation"""
        if not date_str or len(date_str) != 6 or not date_str.isdigit():
            return ""
        try:
            year = int(date_str[:2])
            year += 2000 if year < 50 else 1900
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            return f"{year}-{month:02d}-{day:02d}"
        except:
            return ""

    # Process first line for passport type
    if lines:
        first_line = lines[0]
        
        # Check two-character codes first
        if len(first_line) >= 2:
            code = first_line[:2]
            result["passport_type"] = PASSPORT_TYPES.get(code, PASSPORT_TYPES.get(first_line[0], "Unknown"))
        
        # Standard TD3 format processing
        if len(lines) == 2 and all(len(line) >= 44 for line in lines):
            line1, line2 = lines
            
            # Document details
            result["issuing_country"] = line1[2:5] if len(line1) >= 5 else ""
            
            # Name processing with better edge case handling
            name_parts = line1[5:].split("<<", 1)
            result["surname"] = name_parts[0].replace("<", " ").strip() if name_parts else ""
            if len(name_parts) > 1:
                result["given_names"] = " ".join(
                    [n.replace("<", " ").strip() 
                     for n in name_parts[1].split("<") if n.strip()]
                )
            
            # Document number and validation
            if len(line2) >= 10:
                result["document_number"] = line2[0:9].replace("<", "")
            
            # Nationality with common OCR error correction
            if len(line2) >= 13:
                result["nationality"] = line2[10:13].replace("0", "O")
            
            # Dates and gender
            if len(line2) >= 20:
                result["birth_date"] = parse_date(line2[13:19])
                result["sex"] = GENDER_MAP.get(line2[20], "")
            
            if len(line2) >= 27:
                result["expiry_date"] = parse_date(line2[21:27])

    return result

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or "personal_image" not in request.files:
        return jsonify({
            "error": "Both passport and personal images are required",
            "message": "Verification failed",
            "verify": False,
            "photo_match": False,
            "passport_type": "Unknown"
        }), 400

    try:
        # Image processing with error handling
        passport_img = cv2.imdecode(
            np.frombuffer(request.files["image"].read(), np.uint8), 
            cv2.IMREAD_COLOR
        )
        personal_img = cv2.imdecode(
            np.frombuffer(request.files["personal_image"].read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        # Object detection with confidence threshold
        results = model(passport_img, conf=0.7)[0]
        detections = {
            model.names[int(box.cls)]: passport_img[
                int(box.xyxy[0][1]):int(box.xyxy[0][3]),
                int(box.xyxy[0][0]):int(box.xyxy[0][2])
            ]
            for box in results.boxes 
            if model.names[int(box.cls)] in REQUIRED_CLASSES
        }

        # Initialize response with default values
        response = {
            "message": "Verification successful",
            "verify": True,
            "photo_match": True,
            "passport_type": "Unknown",
            "mrz": {
                "birth_date": "",
                "expiry_date": "",
                "given_names": "",
                "issuing_country": "",
                "nationality": "",
                "document_number": "",
                "raw": "",
                "sex": "",
                "surname": ""
            }
        }

        # MRZ processing if detected
        if "MRZ" in detections:
            mrz_img = detections["MRZ"]
            
            # Advanced image preprocessing
            gray = cv2.cvtColor(mrz_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            gray = cv2.medianBlur(gray, 3)
            _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR with optimized configuration
            mrz_text = pytesseract.image_to_string(
                threshold,
                lang='mrz',
                config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
            ).strip()
            
            # Parse and validate MRZ data
            parsed_data = parse_mrz(mrz_text)
            response.update({
                "passport_type": parsed_data["passport_type"],
                "mrz": {k: v for k, v in parsed_data.items() if k != "passport_type"}
            })

            # Internal debug image saving
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"mrz_{ts}.jpg"), mrz_img)

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Processing error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": "Verification failed",
            "verify": False,
            "photo_match": False,
            "passport_type": "Unknown"
        }), 500

@app.route('/output/<filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)