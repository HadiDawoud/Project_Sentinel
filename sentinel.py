#!/usr/bin/env python3
import argparse
import csv
import io
import json
import os
import signal
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path

from sentinel.pipeline import SentinelPipeline


def signal_handler(signum, frame):
    print("\nInterrupted. Exiting gracefully.", file=sys.stderr)
    sys.exit(130)


signal.signal(signal.SIGINT, signal_handler)

__version__ = "1.0.0"

DEFAULT_CONFIG = os.environ.get('SENTINEL_CONFIG', 'config.yaml')
MAX_INPUT_LENGTH = 10000

LABEL_COLORS = {
    'Non-Radical': '\033[92m',
    'Mildly Radical': '\033[93m',
    'Moderately Radical': '\033[33m',
    'Highly Radical': '\033[91m',
}
RESET_COLOR = '\033[0m'
AVAILABLE_LABELS = ['Non-Radical', 'Mildly Radical', 'Moderately Radical', 'Highly Radical']


def colorize_label(label):
    color = LABEL_COLORS.get(label, '')
    return f"{color}{label}{RESET_COLOR}" if color else label


def to_csv(results):
    output = io.StringIO()
    if not results:
        return ""
    fieldnames = ['label', 'confidence', 'risk_score', 'flagged_terms', 'reasoning']
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for r in results:
        r['flagged_terms'] = ', '.join(r.get('flagged_terms', []))
        writer.writerow(r)
    return output.getvalue()


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
    parser.add_argument(
        '--list-labels',
        action='store_true',
        help="List available classification labels"
    )
    parser.add_argument(
        '--format',
        choices=['json', 'csv', 'yaml'],
        default='json',
        help="Output format (default: json)"
    )

    args = parser.parse_args()

    if args.list_labels:
        print("Available labels:")
        for i, label in enumerate(AVAILABLE_LABELS):
            print(f"  {i}: {colorize_label(label)}")
        sys.exit(0)

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

    if len(args.input) > MAX_INPUT_LENGTH:
        print(f"Warning: Input truncated from {len(args.input)} to {MAX_INPUT_LENGTH} chars", file=sys.stderr)
        args.input = args.input[:MAX_INPUT_LENGTH]

    pipeline = SentinelPipeline(config_path=args.config)

    if Path(args.input).is_file():
        results = pipeline.classify_from_file(args.input, args.output)
        results['timestamp'] = datetime.now(timezone.utc).isoformat()
        if args.label_only:
            for r in results:
                print(colorize_label(r.get('label', 'Unknown')))
        elif args.format == 'csv':
            print(to_csv(results))
        elif args.format == 'yaml':
            print(yaml.dump(results, default_flow_style=False))
        elif not args.quiet:
            print(json.dumps(results, indent=2))
        else:
            print(json.dumps(results))
    else:
        result = pipeline.classify(args.input, return_raw=args.raw)
        result['timestamp'] = datetime.now(timezone.utc).isoformat()
        if args.label_only:
            print(colorize_label(result.get('label', 'Unknown')))
        elif args.format == 'csv':
            print(to_csv([result]))
        elif args.format == 'yaml':
            print(yaml.dump(result, default_flow_style=False))
        elif not args.quiet:
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result))


if __name__ == "__main__":
    main()
