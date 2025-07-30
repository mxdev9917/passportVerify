from flask import Blueprint, request
from controllers.personal_controller import personal_controller

personal_bp = Blueprint("personal", __name__)

@personal_bp.route("/personal/verification", methods=["POST"])
def personal_route():
    return personal_controller(request)
