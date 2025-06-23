from flask import Flask
import subprocess

app = Flask(__name__)

@app.route('/')
def run_script():
    try:
        result = subprocess.run(['python', 'jsonexportsimple.py'], capture_output=True, text=True)
        if result.returncode == 0:
            return f"<pre>✅ Script ran successfully:\n{result.stdout}</pre>"
        else:
            return f"<pre>❌ Script error:\n{result.stderr}</pre>", 500
    except Exception as e:
        return f"<pre>⚠️ Exception: {str(e)}</pre>", 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
