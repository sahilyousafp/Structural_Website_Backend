from flask import Flask
import subprocess
import os
import sys

app = Flask(__name__)

# Run the script in background when Flask app starts
try:
    script_path = os.path.join(os.path.dirname(__file__), 'jsonexportsimplesupabasetake3.py')
    subprocess.Popen([sys.executable, script_path])
except Exception as e:
    print(f"Failed to start background script: {e}")

if __name__ == '__main__':
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"⚠️ Exception: {e}")
