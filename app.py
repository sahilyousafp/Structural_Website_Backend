from flask import Flask
import subprocess
import os
import sys
from flask import jsonify

app = Flask(__name__)

@app.route('/')
def run_script():
    try:
        # build full path to your script
        script_path = os.path.join(os.path.dirname(__file__), 'jsonexportsimple.py')
        # use the same Python interpreter that’s running Flask
        result = subprocess.run(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            return f"<pre>✅ Script ran successfully:\n{result.stdout}</pre>"
        else:
            return f"<pre>❌ Script error:\n{result.stderr}</pre>", 500
    except Exception as e:
        return f"<pre>⚠️ Exception: {e}</pre>", 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
