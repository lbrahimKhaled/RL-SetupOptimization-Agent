"""CLI for read-only inspection of a running AC Gym server."""
import argparse
import json
import os

from .management import ConnectionConfig, inspect_server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inspect = commands.add_parser('inspect', help='read configuration, car/track identity and optional geometry')
    inspect.add_argument('--host', help='Lenovo hostname or LAN IP; overrides AC_HOST')
    inspect.add_argument('--port', type=int, default=2347, help='management TCP port (default: 2347)')
    inspect.add_argument('--timeout', type=float, default=5., help='deadline per query in seconds (default: 5)')
    inspect.add_argument('--include-track', action='store_true', help='also export the complete track geometry dictionary')
    args = parser.parse_args(argv)
    host = args.host if args.host is not None else os.environ.get('AC_HOST')
    try:
        config = ConnectionConfig(host=host, port=args.port, timeout_s=args.timeout)
    except ValueError as exc:
        parser.error(str(exc))
    report = inspect_server(config, args.include_track)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
