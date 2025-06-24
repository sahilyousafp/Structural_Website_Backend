from flask import Flask
import subprocess
import os
import sys

app = Flask(__name__)

# Run the script in background when Flask app starts
try:
<<<<<<< HEAD
    script_path = os.path.join(os.path.dirname(__file__), 'jsonexportsimplesupabasetake3.py')
=======
    script_path = os.path.join(os.path.dirname(__file__), 'jsonexportsimple.py')
>>>>>>> 44d443df4832f2bde504b6f3e8a4a6c64c89d8d7
    subprocess.Popen([sys.executable, script_path])
except Exception as e:
    print(f"Failed to start background script: {e}")

if __name__ == '__main__':
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"⚠️ Exception: {e}")
