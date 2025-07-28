import cv2
import numpy as np
from flask import jsonify
from models.yolo_model import model, REQUIRED_CLASSES
from utils.face_utils import compare_faces
from utils.mrz_parser import parse_mrz, pytesseract

def passport_controller(request):
    if "passport" not in request.files or "photo" not in request.files:
        return jsonify({
            "error": "Both passport and photo files are required",
            "verified": False,
            "score": 0
        }), 400

    try:
        passport_img = cv2.imdecode(np.frombuffer(request.files["passport"].read(), np.uint8), cv2.IMREAD_COLOR)
        user_photo = cv2.imdecode(np.frombuffer(request.files["photo"].read(), np.uint8), cv2.IMREAD_COLOR)

        score = 0
        components = {
            "passport_detected": False,
            "mrz_detected": False,
            "mrz_parsed": False,
            "photo_detected": False,
            "face_detected": False,
            "face_match": False
        }

        results = model(passport_img, conf=0.7)[0]
        detections = {
            model.names[int(box.cls)]: passport_img[
                int(box.xyxy[0][1]):int(box.xyxy[0][3]),
                int(box.xyxy[0][0]):int(box.xyxy[0][2])
            ]
            for box in results.boxes
            if model.names[int(box.cls)] in REQUIRED_CLASSES
        }

        if "Passport" in detections:
            score += 20
            components["passport_detected"] = True

        passport_photo = detections.get("Photo")
        if passport_photo is not None:
            score += 20
            components["photo_detected"] = True

        if user_photo is not None and len(user_photo.shape) == 3:
            score += 10
            components["face_detected"] = True

        similarity = 0.0
        if passport_photo is not None and components["face_detected"]:
            match, similarity = compare_faces(passport_photo, user_photo)
            if match:
                score += 40
                components["face_match"] = True

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
        return jsonify({
            "error": f"Verification failed: {str(e)}",
            "verified": False,
            "score": 0
        }), 500
