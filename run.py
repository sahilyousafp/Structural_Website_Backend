import argparse
import json
from app import create_app
from app.services import generate_structure
from app.config import SUPABASE_URL, SUPABASE_KEY, SUPABASE_MODEL_BUCKET, SUPABASE_MODEL_KEY
import tempfile


def main():
    parser = argparse.ArgumentParser(
        description="Run Flask server or generate structure JSON via CLI."
    )
    parser.add_argument(
        "--generate",
        "-g",
        type=int,
        metavar="FLOORS",
        help="Generate structure JSON for given floor count and exit",
    )
    args = parser.parse_args()
    if args.generate:
        # CLI generation path: download Rhino model from Supabase
        # dynamically import supabase client
        supabase_mod = __import__('supabase')
        create_client = getattr(supabase_mod, 'create_client')
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        model_storage = sb.storage.from_(SUPABASE_MODEL_BUCKET)
        model_data = model_storage.download(SUPABASE_MODEL_KEY)
        tmp = tempfile.NamedTemporaryFile(suffix='.3dm', delete=False)
        tmp.write(model_data)
        tmp.close()
        result = generate_structure(tmp.name, args.generate)
        print(json.dumps(result, indent=2))
    else:
        # Start Flask server
        app = create_app()
        app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
