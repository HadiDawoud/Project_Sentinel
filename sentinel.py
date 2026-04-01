#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from sentinel.pipeline import SentinelPipeline

__version__ = "1.0.0"

DEFAULT_CONFIG = os.environ.get('SENTINEL_CONFIG', 'config.yaml')


def main():
    parser = argparse.ArgumentParser(
        description="Project Sentinel - Radical Content Detection System"
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    parser.add_argument(
        'input',
        nargs='?',
        help="Text to classify, path to file, or - for stdin"
    )
    parser.add_argument(
        '-c', '--config',
        default=DEFAULT_CONFIG,
        help=f"Path to config file (default: {DEFAULT_CONFIG})"
    )
    parser.add_argument(
        '-o', '--output',
        help="Output file path for batch results"
    )
    parser.add_argument(
        '--raw',
        action='store_true',
        help="Include raw intermediate results"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Verbose output"
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help="Minimal output (just the JSON)"
    )
    parser.add_argument(
        '--examples',
        action='store_true',
        help="Show usage examples"
    )
    parser.add_argument(
        '-l', '--label-only',
        action='store_true',
        help="Output just the label (good for piping)"
    )

    args = parser.parse_args()

    if args.examples:
        print("""Usage examples:

  # Classify a single text
  python sentinel.py "some text to classify"

  # Classify from a file
  python sentinel.py input.json -o results.json

  # Read from stdin (piping)
  echo "text here" | python sentinel.py -

  # Use a different config
  python sentinel.py "text" -c custom_config.yaml

  # Minimal output (good for scripts)
  python sentinel.py "text" -q
""")
        sys.exit(0)

    if not args.input:
        print("Error: Input text or file path required", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    if args.input == '-':
        args.input = sys.stdin.read()

    if not args.input.strip():
        print("Error: Input cannot be empty or whitespace only", file=sys.stderr)
        sys.exit(1)

    pipeline = SentinelPipeline(config_path=args.config)

    if Path(args.input).is_file():
        results = pipeline.classify_from_file(args.input, args.output)
        results['timestamp'] = datetime.now(timezone.utc).isoformat()
        if args.label_only:
            for r in results:
                print(r.get('label', 'Unknown'))
        elif not args.quiet:
            print(json.dumps(results, indent=2))
        else:
            print(json.dumps(results))
    else:
        result = pipeline.classify(args.input, return_raw=args.raw)
        result['timestamp'] = datetime.now(timezone.utc).isoformat()
        if args.label_only:
            print(result.get('label', 'Unknown'))
        elif not args.quiet:
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result))


if __name__ == "__main__":
    main()
