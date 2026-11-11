"""Desktop launcher: one process serving the VOIX API and studio UI.

This is what PyInstaller packages into VOIX.exe. It can also be run directly:
    python voix_launcher.py [--host 0.0.0.0] [--port 8000] [--no-browser]
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path


def _base_dirs() -> tuple[Path, Path]:
    """(folder holding bundled resources, folder where user data should live)."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS")), Path(sys.executable).resolve().parent
    here = Path(__file__).resolve().parent
    return here.parent, here


def _free_port(host: str, wanted: int) -> int:
    for port in range(wanted, wanted + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise SystemExit(f"No free port between {wanted} and {wanted + 19}.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="VOIX", description="VOIX - Text in. Voice out.")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to serve your local network")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    resources, home = _base_dirs()
    os.environ.setdefault("VOIX_DATA_DIR", str(home / "voix-data"))
    frontend = resources / "frontend" / ("" if getattr(sys, "frozen", False) else "dist")
    os.environ.setdefault("VOIX_FRONTEND_DIR", str(frontend))

    port = _free_port(args.host, args.port)
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{port}"

    print("VOIX - Text in. Voice out.")
    print(f"  Studio : {url}")
    print(f"  API    : {url}/api/docs")
    print(f"  Data   : {os.environ['VOIX_DATA_DIR']}")
    if args.host not in ("127.0.0.1", "localhost") and not os.environ.get("VOIX_API_KEY"):
        print("  WARNING: serving the network without VOIX_API_KEY. Anyone on it can upload and delete models.")
    print("  Close this window or press Ctrl+C to stop.\n")

    import uvicorn

    from app.main import create_app

    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(), host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
