from flask import Flask
from routes.passort_route import passport_bp
from routes.analyze_route import analyze_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(passport_bp, url_prefix="/api")
app.register_blueprint(analyze_bp, url_prefix="/api")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
