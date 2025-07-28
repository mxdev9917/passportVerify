from flask import jsonify
import cv2
import numpy as np

def analyze_passport_controller(request):
    if "passport" not in request.files:
        return jsonify({"error": "Passport image required"}), 400

    try:
        passport_img = cv2.imdecode(np.frombuffer(request.files["passport"].read(), np.uint8), cv2.IMREAD_COLOR)
        height, width = passport_img.shape[:2]
        return jsonify({
            "message": "Image analyzed successfully",
            "width": width,
            "height": height
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
