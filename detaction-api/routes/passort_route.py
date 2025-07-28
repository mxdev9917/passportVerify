from flask import Blueprint, request
from controllers.passport_controller import passport_controller

passport_bp = Blueprint("passport", __name__)

@passport_bp.route("/verify", methods=["POST"])
def passport_route():
    return passport_controller(request)
