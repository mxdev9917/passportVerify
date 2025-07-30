import io
import os
import uuid
from flask import jsonify, url_for
from PIL import Image
from ultralytics import YOLO

# Load the YOLO model once globally
model = YOLO('personal.pt')
names = model.names  # class names mapping

# Directory to save cropped verified images inside Flask static folder
SAVE_DIR = "static/images"
os.makedirs(SAVE_DIR, exist_ok=True)

def verify_personal_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = model(img)
        detections = results[0].boxes.data  # tensor: [x1, y1, x2, y2, conf, cls]

        for det in detections:
            conf = float(det[4])
            cls_idx = int(det[5])
            label = names[cls_idx]

            if label == 'personal' and conf * 100 >= 80:
                # Crop the detected bounding box area from the image
                x1, y1, x2, y2 = map(int, det[:4].tolist())
                cropped_img = img.crop((x1, y1, x2, y2))
                return True, conf * 100, cropped_img

        # If no "personal" class with required confidence detected, return full image
        return False, 0, img

    except Exception as e:
        return False, 0, str(e)

def personal_controller(request):
    if 'personal_img' not in request.files:
        return jsonify({"error": "No personal_img part"}), 400

    file = request.files['personal_img']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()
        verified, confidence, img_or_error = verify_personal_image(image_bytes)

        if not verified:
            return jsonify({
                "success": False,
                "confidence": round(confidence, 2),
                "message": "Personal verification failed"
            }), 400

        # Save the cropped image with a unique filename
        filename = f"{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(SAVE_DIR, filename)
        img_or_error.save(save_path, format='JPEG')

        # Generate public URL for the saved image
        image_url = url_for('static', filename=f'images/{filename}', _external=True)

        return jsonify({
            "success": True,
            "confidence": round(confidence, 2),
            "image": image_url,
            "message": "Verification successful"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
