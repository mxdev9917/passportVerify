from flask import Flask
from routes.passort_route import passort_bp
from routes.analyze_route import analyze_bp

app = Flask(__name__)

# Register all blueprints
app.register_blueprint(passort_bp)
app.register_blueprint(analyze_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
