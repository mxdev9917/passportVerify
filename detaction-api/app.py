from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime
import pytesseract
import face_recognition

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"  # Adjust path if needed

app = Flask(__name__)

model = YOLO("best.pt")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = model.names
REQUIRED_CLASSES = ["Passport", "Photo", "MRZ"]
CONF_THRESHOLD = 0.8


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
            if 1.3 < aspect_ratio < 1.8 and area > max_area:
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
    lines = mrz_text.strip().splitlines()
    if len(lines) < 2:
        return None

    line1 = lines[0].strip().replace(" ", "")
    line2 = lines[1].strip().replace(" ", "")

    result = {}
    try:
        if line1.startswith("P<"):
            result["country"] = line1[2:5]
            names = line1[5:].split("<<", 1)
            result["last_name"] = names[0].replace("<", " ").strip()
            result["first_name"] = names[1].replace("<", " ").strip() if len(names) > 1 else ""

        passport_number = line2[0:9]
        passport_number_chk = line2[9]
        nationality = line2[10:13]
        birth_date_raw = line2[13:19]
        birth_date_chk = line2[19]
        sex = line2[20]
        expiry_date_raw = line2[21:27]
        expiry_date_chk = line2[27]

        passport_valid = calculate_checksum(passport_number) == int(passport_number_chk)
        birth_valid = calculate_checksum(birth_date_raw) == int(birth_date_chk)
        expiry_valid = calculate_checksum(expiry_date_raw) == int(expiry_date_chk)

        result.update({
            "passport_number": passport_number.replace("<", ""),
            "passport_checksum_valid": bool(passport_valid),
            "nationality": nationality,
            "birth_date": convert_date(birth_date_raw),
            "birth_date_checksum_valid": bool(birth_valid),
            "sex": sex,
            "expiry_date": convert_date(expiry_date_raw),
            "expiry_date_checksum_valid": bool(expiry_valid),
            "mrz_valid": bool(passport_valid and birth_valid and expiry_valid)
        })
    except Exception as e:
        print("MRZ parsing error:", e)
        result["error"] = str(e)

    return result


def compare_faces(photo_img, personal_img):
    try:
        photo_encoding = face_recognition.face_encodings(photo_img)
        personal_encoding = face_recognition.face_encodings(personal_img)

        if not photo_encoding or not personal_encoding:
            print("Face encodings not found in one of the images.")
            return False

        return bool(face_recognition.compare_faces([photo_encoding[0]], personal_encoding[0])[0])
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
                    gray = cv2.cvtColor(mrz_crop, cv2.COLOR_BGR2GRAY)
                    mrz_text = pytesseract.image_to_string(
                        gray,
                        config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
                    ).strip()
                    mrz_data = parse_mrz(mrz_text)

                elif class_name == "Photo":
                    photo_crop = passport_img[y1:y2, x1:x2].copy()

        if all(cls in detected_classes for cls in REQUIRED_CLASSES):
            if mrz_data is None or not mrz_data.get("mrz_valid", False):
                return jsonify({
                    "verify": False,
                    "message": "MRZ verification failed",
                    "reason": "MRZ data invalid or checksum mismatch"
                }), 400

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
                "photo_match": bool(photo_match)
            }

            if passport_filename:
                response["passport_crop_url"] = f"/output/{passport_filename}"

            return jsonify(response)

        else:
            return jsonify({
                "verify": False,
                "message": "Detection failed",
                "reason": "Not all required classes detected with confidence >= 0.8",
                "required_classes": REQUIRED_CLASSES
            }), 400
    except Exception as e:
        print("Unhandled server error:", str(e))
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@app.route("/output/<filename>")
def get_output_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
