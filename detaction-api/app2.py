from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime
import pytesseract
import face_recognition
import traceback

# Configure Tesseract with custom model path
pytesseract.pytesseract.tesseract_cmd = "tesseract"
os.environ['TESSDATA_PREFIX'] = './tessdata'  # Make sure your tessdata folder exists and contains mrz.traineddata

app = Flask(__name__)

# Load YOLO model
model = YOLO("best.pt")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("tessdata", exist_ok=True)  # Ensure tessdata directory exists

CLASS_NAMES = model.names
REQUIRED_CLASSES = ["Passport", "Photo", "MRZ"]
CONF_THRESHOLD = 0.7


def preprocess_for_ocr(image):
    margin = 10
    image = cv2.copyMakeBorder(image, margin, margin, margin, margin,
                               cv2.BORDER_CONSTANT, value=[255, 255, 255])
    scale = 3.0
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    cleaned = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel2)
    return cleaned


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
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        print(f"Warp successful: output size {max_width}x{max_height}")
        return warped
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
    if not mrz_text or len(mrz_text.strip()) < 20:
        print(f"MRZ text too short or empty: '{mrz_text}'")
        return None
    lines = [line.strip().replace(" ", "").upper() for line in mrz_text.strip().splitlines() if len(line.strip()) >= 20]
    print(f"MRZ lines after processing: {lines}")
    if len(lines) < 2:
        print(f"Not enough MRZ lines: {len(lines)}")
        return None
    line1 = lines[0]
    line2 = lines[1]
    if len(line2) < 30:
        print(f"Line 2 too short: {len(line2)} chars")
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
        passport_number_chk = line2[9] if len(line2) > 9 else ""
        nationality = line2[10:13] if len(line2) > 12 else ""
        birth_date_raw = line2[13:19] if len(line2) > 18 else ""
        birth_date_chk = line2[19] if len(line2) > 19 else ""
        sex = line2[20] if len(line2) > 20 else ""
        expiry_date_raw = line2[21:27] if len(line2) > 26 else ""
        expiry_date_chk = line2[27] if len(line2) > 27 else ""
        passport_valid = False
        birth_valid = False
        expiry_valid = False
        if passport_number and passport_number_chk.isdigit():
            passport_valid = calculate_checksum(passport_number) == int(passport_number_chk)
        if birth_date_raw and birth_date_chk.isdigit():
            birth_valid = calculate_checksum(birth_date_raw) == int(birth_date_chk)
        if expiry_date_raw and expiry_date_chk.isdigit():
            expiry_valid = calculate_checksum(expiry_date_raw) == int(expiry_date_chk)
        result.update({
            "passport_number": passport_number.replace("<", ""),
            "passport_checksum_valid": passport_valid,
            "nationality": nationality,
            "birth_date": convert_date(birth_date_raw) if birth_date_raw else "",
            "birth_date_checksum_valid": birth_valid,
            "sex": sex,
            "expiry_date": convert_date(expiry_date_raw) if expiry_date_raw else "",
            "expiry_date_checksum_valid": expiry_valid,
            "mrz_valid": passport_valid and birth_valid and expiry_valid
        })
        print(f"MRZ parsing successful: {result}")
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
            print("One or both images could not be decoded.")
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

            # Validate bbox coords
            if x2 <= x1 or y2 <= y1:
                print(f"Invalid bbox for {class_name}: ({x1},{y1},{x2},{y2})")
                continue
            print(f"Detected {class_name} with confidence {conf:.2f} at bbox ({x1},{y1},{x2},{y2})")

            if class_name == "Passport":
                cropped = passport_img[y1:y2, x1:x2].copy()
                corners = find_passport_corners(cropped)
                passport_crop = warp_to_quadrilateral(cropped, corners)
            elif class_name == "MRZ":
                mrz_crop = passport_img[y1:y2, x1:x2].copy()
                processed_mrz = preprocess_for_ocr(mrz_crop)
                debug_img_path = os.path.join(OUTPUT_DIR, f"mrz_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(debug_img_path, processed_mrz)
                print(f"Saved MRZ preprocessed image at {debug_img_path}")

                # Try OCR with mrz lang first
                mrz_text = ""
                for lang, psm in [("mrz", 6), ("eng", 6), ("eng", 7), ("eng", 8), ("eng", 13)]:
                    try:
                        mrz_text_candidate = pytesseract.image_to_string(
                            processed_mrz,
                            lang=lang,
                            config=f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
                        ).strip()
                        print(f"OCR attempt lang={lang}, psm={psm} output:\n{mrz_text_candidate}")
                        if mrz_text_candidate and len(mrz_text_candidate) >= 20:
                            mrz_text = mrz_text_candidate
                            break
                    except Exception as ocr_e:
                        print(f"OCR error with lang={lang}, psm={psm}: {ocr_e}")

                if mrz_text:
                    lines = [line.strip() for line in mrz_text.splitlines() if len(line.strip()) >= 20]
                    mrz_text = "\n".join(lines)
                    print(f"Filtered MRZ OCR lines:\n{mrz_text}")
                else:
                    print("MRZ OCR returned empty or too short text.")

                mrz_data = parse_mrz(mrz_text)

            elif class_name == "Photo":
                photo_crop = passport_img[y1:y2, x1:x2].copy()

            detected_classes[class_name] = True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if mrz_crop is not None:
            debug_path = os.path.join(OUTPUT_DIR, f"mrz_debug_{timestamp}.jpg")
            cv2.imwrite(debug_path, preprocess_for_ocr(mrz_crop))

        if all(cls in detected_classes for cls in REQUIRED_CLASSES) and mrz_data is not None:
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
            missing_classes = [cls for cls in REQUIRED_CLASSES if cls not in detected_classes]
            reason_parts = []
            if missing_classes:
                reason_parts.append(f"Missing required classes: {', '.join(missing_classes)}")
            if "MRZ" in detected_classes and mrz_data is None:
                reason_parts.append("MRZ text could not be parsed")
            reason = "; ".join(reason_parts) if reason_parts else "Unknown verification failure"
            return jsonify({
                "verify": False,
                "message": "Detection failed",
                "reason": reason,
                "debug": {
                    "mrz_image": f"/output/mrz_debug_{timestamp}.jpg" if mrz_crop is not None else None,
                    "mrz_text": mrz_text or "",
                    "detected_classes": list(detected_classes.keys())
                }
            }), 400
    except Exception as e:
        print("Unhandled server error:", str(e))
        print(traceback.format_exc())
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

            if x2 <= x1 or y2 <= y1:
                print(f"Invalid bbox for {class_name}: ({x1},{y1},{x2},{y2})")
                continue
            print(f"Detected {class_name} with confidence {conf:.2f} at bbox ({x1},{y1},{x2},{y2})")

            if class_name == "MRZ":
                mrz_crop = passport_img[y1:y2, x1:x2].copy()
                processed_mrz = preprocess_for_ocr(mrz_crop)

                debug_img_path = os.path.join(OUTPUT_DIR, f"mrz_verify_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(debug_img_path, processed_mrz)
                print(f"Saved MRZ preprocessed image at {debug_img_path}")

                mrz_text = ""
                for lang, psm in [("mrz", 6), ("eng", 6), ("eng", 7), ("eng", 8), ("eng", 13)]:
                    try:
                        mrz_text_candidate = pytesseract.image_to_string(
                            processed_mrz,
                            lang=lang,
                            config=f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
                        ).strip()
                        print(f"OCR attempt lang={lang}, psm={psm} output:\n{mrz_text_candidate}")
                        if mrz_text_candidate and len(mrz_text_candidate) >= 20:
                            mrz_text = mrz_text_candidate
                            break
                    except Exception as ocr_e:
                        print(f"OCR error with lang={lang}, psm={psm}: {ocr_e}")

                if mrz_text:
                    lines = [line.strip() for line in mrz_text.splitlines() if len(line.strip()) >= 20]
                    mrz_text = "\n".join(lines)
                    print(f"Filtered MRZ OCR lines:\n{mrz_text}")
                else:
                    print("MRZ OCR returned empty or too short text.")

                mrz_data = parse_mrz(mrz_text)

            detected_classes[class_name] = True

        if "MRZ" in detected_classes and mrz_data is not None:
            return jsonify({
                "verify": True,
                "message": "Success",
                "mrz": mrz_data,
                "mrz_text": mrz_text or "",
            })
        else:
            reason = "MRZ not detected or MRZ parsing failed"
            return jsonify({
                "verify": False,
                "message": "Detection failed",
                "reason": reason,
                "debug": {
                    "mrz_text": mrz_text or "",
                    "detected_classes": list(detected_classes.keys())
                }
            }), 400

    except Exception as e:
        print("Unhandled server error in /verify:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": "Internal Server Error", "details": str(e), "verify": False}), 500


@app.route('/output/<filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
