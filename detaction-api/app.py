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
CONF_THRESHOLD = 0.7


def preprocess_for_ocr(image):
    margin = 20
    image = cv2.copyMakeBorder(image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    scale = 2.0
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 15
    )
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return closed


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
            if 1.2 < aspect_ratio < 2.0 and area > max_area:
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
    line1 = lines[0].strip().replace(" ", "").upper()
    line2 = lines[1].strip().replace(" ", "").upper()
    if len(line2) < 30:
        return None
    result = {}
    try:
        if line1.startswith(("P<", "P ")):
            result["document_type"] = "P"
            result["country"] = line1[2:5]
            names = line1[5:].split("<<", 1)
            result["last_name"] = names[0].replace("<", " ").strip()
            result["first_name"] = names[1].replace("<", " ").strip() if len(names) > 1 else ""
        passport_number = line2[:9]
        passport_number_chk = line2[9]
        nationality = line2[10:13]
        birth_date_raw = line2[13:19]
        birth_date_chk = line2[19]
        sex = line2[20]
        expiry_date_raw = line2[21:27]
        expiry_date_chk = line2[27]
        passport_valid = calculate_checksum(passport_number) == int(passport_number_chk) if passport_number_chk.isdigit() else False
        birth_valid = calculate_checksum(birth_date_raw) == int(birth_date_chk) if birth_date_chk.isdigit() else False
        expiry_valid = calculate_checksum(expiry_date_raw) == int(expiry_date_chk) if expiry_date_chk.isdigit() else False
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
        photo_rgb = cv2.cvtColor(photo_img, cv2.COLOR_BGR2RGB)
        personal_rgb = cv2.cvtColor(personal_img, cv2.COLOR_BGR2RGB)
        photo_encoding = face_recognition.face_encodings(photo_rgb)
        personal_encoding = face_recognition.face_encodings(personal_rgb)
        if not photo_encoding or not personal_encoding:
            print("Face encodings not found.")
            return False
        matches = face_recognition.compare_faces([photo_encoding[0]], personal_encoding[0], tolerance=0.6)
        return bool(matches[0])
    except Exception as e:
        print("Face comparison error:", e)
        return False


@app.route("/help", methods=["GET"])
def help():
    """Provides documentation and usage instructions for the API"""
    help_info = {
        "service": "Passport Verification API",
        "description": "This service verifies passport documents by detecting key elements (Passport, Photo, MRZ) and extracting information from the Machine Readable Zone (MRZ). It can also compare the passport photo with a personal photo for identity verification.",
        "endpoints": [
            {
                "path": "/predict",
                "method": "POST",
                "description": "Full passport verification including face matching",
                "parameters": {
                    "image": "Passport image file (required)",
                    "personal_image": "Personal photo for face comparison (required)"
                },
                "response": {
                    "verify": "Boolean indicating overall verification status",
                    "message": "Status message",
                    "mrz": {
                        "birth_date": "Extracted birth date",
                        "birth_date_valid": "Checksum validation status",
                        "expiry_date": "Extracted expiry date",
                        "expiry_date_valid": "Checksum validation status",
                        "nationality": "Extracted nationality code",
                        "passport_number": "Extracted passport number",
                        "passport_number_valid": "Checksum validation status",
                        "raw_text": "Raw MRZ text",
                        "sex": "Extracted gender"
                    },
                    "photo_match": "Boolean indicating if passport photo matches personal photo",
                    "debug": "Debug information including processed images"
                }
            },
            {
                "path": "/verify",
                "method": "POST",
                "description": "Basic passport verification (MRZ and document validation only)",
                "parameters": {
                    "image": "Passport image file (required)"
                },
                "response": {
                    "verify": "Boolean indicating verification status",
                    "message": "Status message",
                    "mrz": "Extracted MRZ information (same format as /predict)",
                    "debug": "Debug information"
                }
            },
            {
                "path": "/output/<filename>",
                "method": "GET",
                "description": "Retrieve processed images for debugging",
                "parameters": {
                    "filename": "Name of the output file"
                }
            },
            {
                "path": "/help",
                "method": "GET",
                "description": "API documentation and usage instructions"
            }
        ],
        "requirements": {
            "passport_image": "Should clearly show the passport with visible MRZ zone",
            "personal_image": "Clear frontal face photo for comparison",
            "supported_formats": "JPEG, PNG"
        },
        "notes": [
            "The service uses computer vision and may not be 100% accurate",
            "Face matching requires reasonably clear images of faces",
            "MRZ parsing requires the MRZ to be clearly visible and properly oriented",
            "Debug images are automatically deleted when the server restarts"
        ],
        "example_usage": {
            "curl_predict": 'curl -X POST -F "image=@passport.jpg" -F "personal_image=@selfie.jpg" http://localhost:5000/predict',
            "curl_verify": 'curl -X POST -F "image=@passport.jpg" http://localhost:5000/verify',
            "curl_help": 'curl http://localhost:5000/help'
        }
    }
    return jsonify(help_info)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files or 'personal_image' not in request.files:
            return jsonify({"error": "Both passport image and personal image are required"}), 400
        passport_file = request.files['image']
        personal_file = request.files['personal_image']
        passport_img = cv2.imdecode(np.frombuffer(passport_file.read(), np.uint8), cv2.IMREAD_COLOR)
        personal_img = cv2.imdecode(np.frombuffer(personal_file.read(), np.uint8), cv2.IMREAD_COLOR)
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
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if class_name == "Passport":
                cropped = passport_img[y1:y2, x1:x2].copy()
                corners = find_passport_corners(cropped)
                passport_crop = warp_to_quadrilateral(cropped, corners)
            elif class_name == "MRZ":
                mrz_crop = passport_img[y1:y2, x1:x2].copy()
                processed_mrz = preprocess_for_ocr(mrz_crop)
                mrz_text = pytesseract.image_to_string(processed_mrz, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789< -c preserve_interword_spaces=1').strip()
                mrz_text = "\n".join([line for line in mrz_text.splitlines() if len(line.strip()) >= 25])
                mrz_data = parse_mrz(mrz_text)
            elif class_name == "Photo":
                photo_crop = passport_img[y1:y2, x1:x2].copy()
            detected_classes[class_name] = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if mrz_crop is not None:
            debug_path = os.path.join(OUTPUT_DIR, f"mrz_debug_{timestamp}.jpg")
            cv2.imwrite(debug_path, preprocess_for_ocr(mrz_crop))
        if all(cls in detected_classes for cls in REQUIRED_CLASSES):
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
                "photo_match": compare_faces(photo_crop, personal_img) if photo_crop is not None else False,
                "debug": {
                    "mrz_image": f"/output/mrz_debug_{timestamp}.jpg"
                }
            }
            if passport_crop is not None:
                passport_filename = f"passport_{timestamp}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, passport_filename), passport_crop)
                response["passport_crop_url"] = f"/output/{passport_filename}"
            return jsonify(response)
        else:
            return jsonify({
                "verify": False,
                "message": "Detection failed",
                "reason": f"Missing required classes: {', '.join([cls for cls in REQUIRED_CLASSES if cls not in detected_classes])}",
                "debug": {
                    "mrz_image": f"/output/mrz_debug_{timestamp}.jpg" if mrz_crop is not None else None,
                    "mrz_text": mrz_text
                }
            }), 400
    except Exception as e:
        print("Unhandled server error:", str(e))
        return jsonify({"error": "Internal Server Error", "details": str(e), "verify": False}), 500


@app.route("/verify", methods=["POST"])
def verify_passport():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Passport image is required"}), 400
        passport_file = request.files['image']
        passport_img = cv2.imdecode(np.frombuffer(passport_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if passport_img is None:
            return jsonify({"error": "Invalid image uploaded"}), 400
        results = model(passport_img)
        boxes = results[0].boxes
        detected_classes = {}
        mrz_crop = None
        mrz_text = None
        mrz_data = None
        for box in boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue
            cls_id = int(box.cls[0])
            class_name = CLASS_NAMES.get(cls_id, "unknown")
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if class_name == "MRZ":
                mrz_crop = passport_img[y1:y2, x1:x2].copy()
                processed_mrz = preprocess_for_ocr(mrz_crop)
                mrz_text = pytesseract.image_to_string(processed_mrz, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789< -c preserve_interword_spaces=1').strip()
                mrz_text = "\n".join([line for line in mrz_text.splitlines() if len(line.strip()) >= 25])
                mrz_data = parse_mrz(mrz_text)
            detected_classes[class_name] = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if mrz_crop is not None:
            debug_path = os.path.join(OUTPUT_DIR, f"mrz_verify_debug_{timestamp}.jpg")
            cv2.imwrite(debug_path, preprocess_for_ocr(mrz_crop))
        if "Passport" in detected_classes and "MRZ" in detected_classes and mrz_data:
            return jsonify({
                "verify": True,
                "message": "Passport verified",
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
                "debug": {
                    "mrz_image": f"/output/mrz_verify_debug_{timestamp}.jpg"
                }
            })
        else:
            return jsonify({
                "verify": False,
                "message": "Verification failed",
                "reason": f"Missing required classes: {', '.join([cls for cls in ['Passport', 'MRZ'] if cls not in detected_classes])}",
                "debug": {
                    "mrz_text": mrz_text or "",
                    "mrz_image": f"/output/mrz_verify_debug_{timestamp}.jpg" if mrz_crop is not None else None
                }
            }), 400
    except Exception as e:
        print("Verification error:", str(e))
        return jsonify({"error": "Internal Server Error", "details": str(e), "verify": False}), 500


@app.route("/output/<filename>")
def get_output_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)