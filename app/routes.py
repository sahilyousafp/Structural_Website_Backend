from flask import Blueprint, request, jsonify
from .services import generate_structure
from .config import SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET, SUPABASE_MODEL_BUCKET, SUPABASE_MODEL_KEY  # noqa: E0401
import json, io
import tempfile
from datetime import datetime

bp = Blueprint('main', __name__)

@bp.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Structural API is running.'}), 200

@bp.route('/generate', methods=['GET','POST'])
def generate():
    # support GET ?num_floors= and POST JSON
    if request.method == 'GET':
        num_floors = request.args.get('num_floors', type=int)
    else:
        data = request.get_json() or {}
        num_floors = data.get('num_floors')
    if not isinstance(num_floors, int) or num_floors < 1:
        return jsonify({'error': 'num_floors must be a positive integer.'}), 400
    try:
        # download Rhino model from Supabase storage
        supabase_mod = __import__('supabase')
        create_client = getattr(supabase_mod, 'create_client')
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        model_storage = sb.storage.from_(SUPABASE_MODEL_BUCKET)
        model_data = model_storage.download(SUPABASE_MODEL_KEY)
        tmp = tempfile.NamedTemporaryFile(suffix='.3dm', delete=False)
        tmp.write(model_data)
        tmp.close()
        # generate using downloaded model
        result = generate_structure(tmp.name, num_floors)
        # upload to Supabase
        # dynamically import supabase client
        create_client = getattr(supabase_mod, 'create_client')
        # reuse sb client
        storage = sb.storage.from_(SUPABASE_BUCKET)
        filename = f"structure_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
        buf = io.BytesIO(json.dumps(result).encode('utf-8'))
        storage.upload(filename, buf, {'content-type': 'application/json'})
        public_url = storage.get_public_url(filename)['publicUrl']
        return jsonify({'result': result, 'url': public_url}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
