from flask import Blueprint, request
from controllers.analyze_controller import analyze_passport_controller

analyze_bp = Blueprint("analyze", __name__)

@analyze_bp.route("/analyze", methods=["POST"])
def analyze_route():
    return analyze_passport_controller(request)
