import os
from flask import Flask, request, jsonify, send_file
import converter
import analyzer

app = Flask(__name__)

@app.route("/convert", methods=["POST"])
def convert():
    data = request.get_json()
    if not data or "file_key" not in data:
        return jsonify({"error": "file_key is required"}), 400
    file_key = data["file_key"]
    try:
        glb_path = converter.convert_to_glb(file_key)
        filename = os.path.splitext(os.path.basename(file_key))[0] + ".glb"
        return send_file(glb_path, mimetype="model/gltf-binary", as_attachment=True, attachment_filename=filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "file_key" not in data or "num_floors" not in data:
        return jsonify({"error": "file_key and num_floors are required"}), 400
    file_key = data["file_key"]
    try:
        num_floors = int(data["num_floors"])
    except ValueError:
        return jsonify({"error": "num_floors must be an integer"}), 400
    try:
        result = analyzer.analyze_structure(file_key, num_floors)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
