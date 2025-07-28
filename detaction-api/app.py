from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract
import face_recognition
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
pytesseract.pytesseract.tesseract_cmd = "tesseract"
model = YOLO("best.pt")
REQUIRED_CLASSES = ["Passport", "Photo", "MRZ"]

app = Flask(__name__)

def parse_mrz(mrz_text):
    """Parse MRZ text into structured data"""
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

    # Clean MRZ text
    cleaned_lines = [''.join(c for c in line if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<') 
                    for line in lines]

    # TD3 format (2 lines)
    if len(cleaned_lines) == 2 and all(len(line) >= 44 for line in cleaned_lines):
        line1, line2 = cleaned_lines
        result.update({
            "issuing_country": line1[2:5],
            "document_number": line2[0:9].replace("<", ""),
            "nationality": line2[10:13].replace("0", "O"),
            "birth_date": parse_date(line2[13:19]),
            "sex": parse_gender(line2[20]),
            "expiry_date": parse_date(line2[21:27])
        })
        
        name_parts = line1[5:].split("<<", 1)
        result["surname"] = name_parts[0].replace("<", " ").strip()
        if len(name_parts) > 1:
            result["given_names"] = " ".join(
                n.replace("<", " ").strip() 
                for n in name_parts[1].split() if n.strip()
            )

    # TD1 format (3 lines)
    elif len(cleaned_lines) == 3 and all(len(line) >= 30 for line in cleaned_lines):
        line1, line2, line3 = cleaned_lines
        result.update({
            "document_number": line1[5:14].replace("<", ""),
            "issuing_country": line1[2:5],
            "birth_date": parse_date(line2[0:6]),
            "sex": parse_gender(line2[7]),
            "expiry_date": parse_date(line2[8:14]),
            "nationality": line2[15:18].replace("0", "O")
        })

        name_parts = line3.split("<<", 1)
        result["surname"] = name_parts[0].replace("<", " ").strip()
        if len(name_parts) > 1:
            result["given_names"] = " ".join(
                n.replace("<", " ").strip() 
                for n in name_parts[1].split() if n.strip()
            )

    return result

def parse_date(date_str):
    """Parse MRZ date format (YYMMDD) to YYYY-MM-DD"""
    if not date_str or len(date_str) != 6 or not date_str.isdigit():
        return ""
    year = int(date_str[:2]) + (2000 if int(date_str[:2]) < 50 else 1900)
    return f"{year}-{date_str[2:4]}-{date_str[4:6]}"

def parse_gender(gender_char):
    """Parse gender character"""
    return {'M': 'Male', 'F': 'Female'}.get(gender_char, "Unspecified")

def compare_faces(img1, img2):
    """Compare two faces and return similarity"""
    try:
        rgb_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        encodings1 = face_recognition.face_encodings(rgb_img1)
        encodings2 = face_recognition.face_encodings(rgb_img2)

        if not encodings1 or not encodings2:
            return False, 0.0

        face_distance = face_recognition.face_distance([encodings1[0]], encodings2[0])[0]
        similarity = 1 - face_distance
        return similarity > 0.5, similarity
        
    except Exception as e:
        app.logger.error(f"Face comparison error: {str(e)}")
        return False, 0.0

@app.route("/verify", methods=["POST"])
def verify_passport():
    """Main verification endpoint"""
    if "passport" not in request.files or "photo" not in request.files:
        return jsonify({
            "error": "Both passport and photo files are required",
            "verified": False,
            "score": 0
        }), 400

    try:
        # Load images directly from memory
        passport_img = cv2.imdecode(
            np.frombuffer(request.files["passport"].read(), np.uint8),
            cv2.IMREAD_COLOR
        )
        user_photo = cv2.imdecode(
            np.frombuffer(request.files["photo"].read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        # Initialize verification metrics
        score = 0
        components = {
            "passport_detected": False,
            "mrz_detected": False,
            "mrz_parsed": False,
            "photo_detected": False,
            "face_detected": False,
            "face_match": False
        }

        # Detect passport elements
        results = model(passport_img, conf=0.7)[0]
        detections = {
            model.names[int(box.cls)]: passport_img[
                int(box.xyxy[0][1]):int(box.xyxy[0][3]),
                int(box.xyxy[0][0]):int(box.xyxy[0][2])
            ]
            for box in results.boxes
            if model.names[int(box.cls)] in REQUIRED_CLASSES
        }

        # Check passport detection
        if "Passport" in detections:
            score += 20
            components["passport_detected"] = True

        # Process passport photo
        passport_photo = detections.get("Photo")
        if passport_photo is not None:
            score += 20
            components["photo_detected"] = True

        # Process user photo
        if face_recognition.face_locations(user_photo):
            score += 10
            components["face_detected"] = True

        # Compare faces if both photos available
        similarity = 0.0
        if passport_photo is not None and components["face_detected"]:
            match, similarity = compare_faces(passport_photo, user_photo)
            if match:
                score += 40
                components["face_match"] = True

        # Process MRZ if available
        mrz_data = {}
        if "MRZ" in detections:
            score += 10
            components["mrz_detected"] = True
            
            mrz_img = detections["MRZ"]
            processed = cv2.cvtColor(mrz_img, cv2.COLOR_BGR2GRAY)
            processed = cv2.resize(processed, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            mrz_text = pytesseract.image_to_string(
                processed,
                lang='mrz',
                config='--psm 6 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
            ).strip()

            mrz_data = parse_mrz(mrz_text)
            if mrz_data.get("document_number"):
                score += 10
                components["mrz_parsed"] = True

        return jsonify({
            "verified": score >= 70,
            "score": score,
            "components": components,
            "similarity": similarity,
            "mrz_data": mrz_data
        })

    except Exception as e:
        app.logger.error(f"Verification failed: {str(e)}")
        return jsonify({
            "error": "Processing error",
            "verified": False,
            "score": 0
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)