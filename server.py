from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({
            "error": "Keine Nachricht erhalten."
        }), 400

    message = data["message"].strip()

    if not message:
        return jsonify({
            "error": "Die Nachricht ist leer."
        }), 400

    # Hier wird später unser eigenes neuronales Netzwerk aufgerufen.
    response = "Ich habe deine Nachricht erhalten: " + message

    return jsonify({
        "response": response
    })


if __name__ == "__main__":
    print("Meine KI läuft auf http://127.0.0.1:5000")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
