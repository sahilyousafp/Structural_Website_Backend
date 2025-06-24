from flask import Flask
import subprocess
import os
import sys

app = Flask(__name__)

def run_background_script():
    """Helper function to run the background script"""
    try:
        script_path = os.path.join(os.path.dirname(__file__), r'./jsonexportsimplesupabasetake3')
        subprocess.Popen([sys.executable, script_path])
        return True
    except Exception as e:
        print(f"Failed to start background script: {e}")
        return False

@app.route('/refresh')
def refresh():
    """Endpoint to refresh/rerun the background script"""
    if run_background_script():
        return "✅ Background script refreshed successfully!"
    else:
        return "❌ Failed to refresh background script", 500

# Run the script in background when Flask app starts
<<<<<<< HEAD
try:
    script_path = os.path.join(os.path.dirname(__file__), 'jsonexportsimple.py')
    subprocess.Popen([sys.executable, script_path])
except Exception as e:
    print(f"Failed to start background script: {e}")
=======
run_background_script()
>>>>>>> 0e1bf0cdcecd22a3c293e32ba95a7043034d465e

if __name__ == '__main__':
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"⚠️ Exception: {e}")
