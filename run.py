import argparse
import json
from app import create_app
from app.config import RHINO_PATH
from app.services import generate_structure


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
        # CLI generation path
        result = generate_structure(RHINO_PATH, args.generate)
        print(json.dumps(result, indent=2))
    else:
        # Start Flask server
        app = create_app()
        app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
