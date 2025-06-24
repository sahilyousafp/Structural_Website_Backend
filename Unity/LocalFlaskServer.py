import os  # Add this missing import
import random
from flask import Flask, request
from Library import Generate

PORT = 8765
STORAGE_FOLDER = "Storage/"

app = Flask(__name__)

# Ensure Storage directory exists
os.makedirs(STORAGE_FOLDER, exist_ok=True)

@app.route('/<path:subpath>', methods=['GET', 'PUT'])
def main(subpath):
    filename = f"{STORAGE_FOLDER}/{subpath}"
    if request.method == 'PUT':
        # Received the file, can process it here
        dataRaw = request.get_data()
        with open(filename, "wb") as f:
            f.write(dataRaw)
        # Fix: Only pass the file path to Generate()
        Generate(glb_file_path=filename)  # Use full path including Storage/
    elif request.method == 'GET':
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                dataRaw = file.read()
            return dataRaw
        else:
            return "File not found", 404
    return ""

if __name__ == '__main__':
    try:
        print(f"Server starting on http://0.0.0.0:{PORT}")
        print(f"Upload files to: http://192.168.0.13:{PORT}/filename.glb")
        app.run(debug=True, port=PORT, host='0.0.0.0')
    except KeyboardInterrupt:
        pass