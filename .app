from flask import Flask, jsonify, request
import json
import os
from supabase import create_client

app = Flask(__name__)

# Local JSON data fallback
DATA_FILE = os.path.join(os.path.dirname(__file__), 'GLB Trial Model.json')

# Supabase configuration
SUPABASE_URL = 'https://apdbfbjnlsxjfubqahtl.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFwZGJmYmpubHN4amZ1YnFhaHRsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc0NzY1MzgsImV4cCI6MjA2MzA1MjUzOH0.DPINyYHHUzcuQ6AOcp8hh1W1eIiamOFPKFRMNfHypSU'
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# Serve latest JSON at root from Supabase storage, fallback to local file
@app.route('/')
def root():
    try:
        # list files in bucket
        files = sb.storage.from_('models').list('79edaed4-a719-4390-a485-519b68fa68ea/')
        if not files:
            raise RuntimeError('No files in Supabase storage')
        # pick the last file in list
        latest = files[-1]['name']
        data_bytes = sb.storage.from_('models').list('79edaed4-a719-4390-a485-519b68fa68ea/').download(latest)
        data = json.loads(data_bytes.decode('utf-8'))
    except Exception:
        # fallback to local file
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            return jsonify({'error': 'Local data file not found.'}), 404
    return jsonify(data)

@app.route('/status')
def status():
    try:
        files = sb.storage.from_('models').list('79edaed4-a719-4390-a485-519b68fa68ea/')
        names = [f['name'] for f in files]
        latest_file = names[-1] if names else None
        has_glb = any(name.lower().endswith('.glb') for name in names)
        has_obj = any(name.lower().endswith('.obj') for name in names)
        return jsonify({
            'supabase_access': True,
            'file_count': len(names),
            'has_glb': has_glb,
            'has_obj': has_obj,
            'latest': latest_file
        })
    except Exception as e:
        return jsonify({'supabase_access': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
