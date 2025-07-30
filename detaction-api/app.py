from flask import Flask
from routes.passort_route import passport_bp
from routes.personal_route import personal_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(passport_bp, url_prefix="/api")
app.register_blueprint(personal_bp, url_prefix="/api")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
