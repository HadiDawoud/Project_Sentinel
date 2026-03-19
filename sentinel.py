#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from sentinel.pipeline import SentinelPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Project Sentinel - Radical Content Detection System"
    )
    parser.add_argument(
        'input',
        nargs='?',
        help="Text to classify or path to input file (JSON, JSONL, TXT)"
    )
    parser.add_argument(
        '-c', '--config',
        default='config.yaml',
        help="Path to config file (default: config.yaml)"
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

    args = parser.parse_args()

    if not args.input:
        print("Error: Input text or file path required")
        parser.print_help()
        sys.exit(1)

    pipeline = SentinelPipeline(config_path=args.config)

    if Path(args.input).is_file():
        results = pipeline.classify_from_file(args.input, args.output)
        print(json.dumps(results, indent=2))
    else:
        result = pipeline.classify(args.input, return_raw=args.raw)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
