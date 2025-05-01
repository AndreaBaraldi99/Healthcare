#!/root/HealthcareApp/.venv/bin/python

from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Autorizza solo il dominio del frontend
CORS(app, origins=["*"], methods=["GET"])

@app.route("/", methods=["GET"])
def hello():
    return jsonify(message="Hello, World from Flask API")

if __name__ == "__main__":
    # Gunicorn in produzione, qui per sviluppo rapido
    app.run(host="0.0.0.0", port=5000)