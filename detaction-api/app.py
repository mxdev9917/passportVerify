from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime
import pytesseract
import face_recognition

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = "tesseract"

app = Flask(__name__)

model = YOLO("best.pt")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = model.names
REQUIRED_CLASSES = ["Passport", "Photo", "MRZ"]
CONF_THRESHOLD = 0.7  # Slightly lowered for better detection

def preprocess_for_ocr(image):
    """Enhanced preprocessing for MRZ OCR"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply dilation to make characters more solid
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    return dilated

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_passport_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_approx = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 1.2 < aspect_ratio < 2.0 and area > max_area:  # More flexible aspect ratio
                max_area = area
                best_approx = approx

    if best_approx is None:
        return None

    pts = best_approx.reshape(4, 2)
    return order_points(pts)

def warp_to_quadrilateral(image, src_pts):
    try:
        if src_pts is None or len(src_pts) != 4:
            return image
            
        width_a = np.linalg.norm(src_pts[2] - src_pts[3])
        width_b = np.linalg.norm(src_pts[1] - src_pts[0])
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(src_pts[1] - src_pts[2])
        height_b = np.linalg.norm(src_pts[0] - src_pts[3])
        max_height = max(int(height_a), int(height_b))

        dst_pts = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(image, M, (max_width, max_height))
    except Exception as e:
        print("Warp error:", e)
        return image

def calculate_checksum(data):
    weights = [7, 3, 1]
    total = 0
    for i, char in enumerate(data):
        if char.isdigit():
            value = int(char)
        elif char.isalpha():
            value = ord(char.upper()) - ord('A') + 10
        elif char == '<':
            value = 0
        else:
            return -1
        total += value * weights[i % 3]
    return total % 10

def convert_date(date_str):
    try:
        year = int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        year += 1900 if year >= 50 else 2000
        return f"{year:04d}-{month:02d}-{day:02d}"
    except:
        return date_str

def parse_mrz(mrz_text):
    lines = [line for line in mrz_text.strip().splitlines() if len(line.strip()) >= 30]
    if len(lines) < 2:
        return None

    # Clean up the lines
    line1 = lines[0].strip().replace(" ", "").upper()
    line2 = lines[1].strip().replace(" ", "").upper()
    
    # Additional validation for MRZ format
    if len(line2) < 30:
        return None

    result = {}
    try:
        # More flexible parsing
        if line1.startswith(("P<", "P ")):
            result["document_type"] = "P"
            result["country"] = line1[2:5]
            names = line1[5:].split("<<", 1)
            result["last_name"] = names[0].replace("<", " ").strip()
            if len(names) > 1:
                result["first_name"] = names[1].replace("<", " ").strip()
            else:
                result["first_name"] = ""
        
        # More tolerant MRZ line2 parsing
        passport_number = line2[:9]
        passport_number_chk = line2[9] if len(line2) > 9 else '<'
        nationality = line2[10:13] if len(line2) > 13 else 'XXX'
        birth_date_raw = line2[13:19] if len(line2) > 19 else '000000'
        birth_date_chk = line2[19] if len(line2) > 19 else '<'
        sex = line2[20] if len(line2) > 20 else '<'
        expiry_date_raw = line2[21:27] if len(line2) > 27 else '000000'
        expiry_date_chk = line2[27] if len(line2) > 27 else '<'

        # Calculate checksums with fallback
        passport_valid = True  # Assume valid by default
        birth_valid = True
        expiry_valid = True
        
        try:
            passport_valid = calculate_checksum(passport_number) == int(passport_number_chk)
        except:
            passport_valid = False
            
        try:
            birth_valid = calculate_checksum(birth_date_raw) == int(birth_date_chk)
        except:
            birth_valid = False
            
        try:
            expiry_valid = calculate_checksum(expiry_date_raw) == int(expiry_date_chk)
        except:
            expiry_valid = False

        result.update({
            "passport_number": passport_number.replace("<", ""),
            "passport_checksum_valid": passport_valid,
            "nationality": nationality,
            "birth_date": convert_date(birth_date_raw),
            "birth_date_checksum_valid": birth_valid,
            "sex": sex,
            "expiry_date": convert_date(expiry_date_raw),
            "expiry_date_checksum_valid": expiry_valid,
            "mrz_valid": passport_valid and birth_valid and expiry_valid
        })
    except Exception as e:
        print("MRZ parsing error:", e)
        result["error"] = str(e)
        result["mrz_valid"] = False

    return result

def compare_faces(photo_img, personal_img):
    try:
        # Convert to RGB (face_recognition uses RGB)
        photo_rgb = cv2.cvtColor(photo_img, cv2.COLOR_BGR2RGB)
        personal_rgb = cv2.cvtColor(personal_img, cv2.COLOR_BGR2RGB)
        
        photo_encoding = face_recognition.face_encodings(photo_rgb)
        personal_encoding = face_recognition.face_encodings(personal_rgb)

        if not photo_encoding or not personal_encoding:
            print("Face encodings not found in one of the images.")
            return False

        # Compare faces with tolerance
        matches = face_recognition.compare_faces(
            [photo_encoding[0]], 
            personal_encoding[0], 
            tolerance=0.6  # Slightly more tolerant
        )
        return bool(matches[0])
    except Exception as e:
        print("Face comparison error:", e)
        return False

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files or 'personal_image' not in request.files:
            return jsonify({"error": "Both passport image and personal image are required"}), 400

        passport_file = request.files['image']
        personal_file = request.files['personal_image']

        passport_bytes = passport_file.read()
        personal_bytes = personal_file.read()

        passport_img = cv2.imdecode(np.frombuffer(passport_bytes, np.uint8), cv2.IMREAD_COLOR)
        personal_img = cv2.imdecode(np.frombuffer(personal_bytes, np.uint8), cv2.IMREAD_COLOR)

        if passport_img is None or personal_img is None:
            return jsonify({"error": "Invalid image(s) uploaded"}), 400

        results = model(passport_img)
        boxes = results[0].boxes

        detected_classes = {}
        passport_crop = None
        mrz_crop = None
        mrz_text = None
        mrz_data = None
        photo_crop = None

        for box in boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            cls_id = int(box.cls[0])
            class_name = CLASS_NAMES.get(cls_id, "unknown")

            if class_name in REQUIRED_CLASSES:
                detected_classes[class_name] = True
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if class_name == "Passport":
                    cropped = passport_img[y1:y2, x1:x2].copy()
                    corners = find_passport_corners(cropped)
                    passport_crop = warp_to_quadrilateral(cropped, corners)

                elif class_name == "MRZ":
                    mrz_crop = passport_img[y1:y2, x1:x2].copy()
                    processed_mrz = preprocess_for_ocr(mrz_crop)
                    mrz_text = pytesseract.image_to_string(
                        processed_mrz,
                        config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789< -c preserve_interword_spaces=1'
                    ).strip()
                    mrz_data = parse_mrz(mrz_text)

                elif class_name == "Photo":
                    photo_crop = passport_img[y1:y2, x1:x2].copy()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Debug output
        if mrz_crop is not None:
            debug_path = os.path.join(OUTPUT_DIR, f"mrz_debug_{timestamp}.jpg")
            cv2.imwrite(debug_path, preprocess_for_ocr(mrz_crop))
            print(f"Saved debug MRZ image to {debug_path}")

        if all(cls in detected_classes for cls in REQUIRED_CLASSES):
            passport_filename = None
            if passport_crop is not None:
                passport_filename = f"passport_{timestamp}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, passport_filename), passport_crop)

            photo_match = False
            if photo_crop is not None:
                photo_match = compare_faces(photo_crop, personal_img)

            response = {
                "verify": True,
                "message": "Success",
                "mrz": {
                    "birth_date": mrz_data.get("birth_date", ""),
                    "birth_date_valid": bool(mrz_data.get("birth_date_checksum_valid", False)),
                    "expiry_date": mrz_data.get("expiry_date", ""),
                    "expiry_date_valid": bool(mrz_data.get("expiry_date_checksum_valid", False)),
                    "nationality": mrz_data.get("nationality", ""),
                    "passport_number": mrz_data.get("passport_number", ""),
                    "passport_number_valid": bool(mrz_data.get("passport_checksum_valid", False)),
                    "raw_text": mrz_text or "",
                    "sex": mrz_data.get("sex", ""),
                },
                "photo_match": bool(photo_match),
                "debug": {
                    "mrz_image": f"/output/mrz_debug_{timestamp}.jpg" if mrz_crop is not None else None
                }
            }

            if passport_filename:
                response["passport_crop_url"] = f"/output/{passport_filename}"

            return jsonify(response)

        else:
            missing_classes = [cls for cls in REQUIRED_CLASSES if cls not in detected_classes]
            return jsonify({
                "verify": False,
                "message": "Detection failed",
                "reason": f"Missing required classes: {', '.join(missing_classes)}",
                "required_classes": REQUIRED_CLASSES,
                "detected_classes": list(detected_classes.keys()),
                "debug": {
                    "mrz_image": f"/output/mrz_debug_{timestamp}.jpg" if mrz_crop is not None else None,
                    "mrz_text": mrz_text
                }
            }), 400
    except Exception as e:
        print("Unhandled server error:", str(e))
        return jsonify({
            "error": "Internal Server Error", 
            "details": str(e),
            "verify": False
        }), 500

@app.route("/output/<filename>")
def get_output_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)