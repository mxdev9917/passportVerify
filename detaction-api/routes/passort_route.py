from flask import Blueprint, request, jsonify
from controllers.passport_controller import passport_controller

passport_bp = Blueprint("passport", __name__)

@passport_bp.route("/verify", methods=["POST"])
def passport_route():
    return passport_controller(request)

# ✅ Add this health check endpoint
@passport_bp.route("/check", methods=["GET"])
def check_route():
    return jsonify({"status": "ok", "message": "API is running"}), 200
