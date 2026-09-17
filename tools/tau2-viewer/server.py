#!/usr/bin/env python3
"""Serve the local tau2 trace viewer and simulation results."""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

VIEWER_DIR = Path(__file__).resolve().parent
TAU2_ROOT = VIEWER_DIR.parents[1]
DEFAULT_SIMULATIONS_DIR = TAU2_ROOT / "data" / "simulations"


def load_results(results_path: Path) -> dict:
    """Load a results file, including split simulation files when present."""
    results = json.loads(results_path.read_text(encoding="utf-8"))
    simulations_dir = results_path.parent / "simulations"
    if simulations_dir.is_dir() and not results.get("simulations"):
        results["simulations"] = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(simulations_dir.glob("*.json"))
        ]
    return results


def reward_of(simulation: dict) -> float | None:
    """Return a simulation reward when it is numeric."""
    reward = (simulation.get("reward_info") or {}).get("reward")
    return float(reward) if isinstance(reward, int | float) else None


def summarize_run(results_path: Path, simulations_root: Path) -> dict:
    """Build the lightweight record used by the run dropdown."""
    results = load_results(results_path)
    simulations = results.get("simulations", [])
    rewards = [
        reward for item in simulations if (reward := reward_of(item)) is not None
    ]
    passed = sum(reward == 1 for reward in rewards)
    relative_dir = results_path.parent.relative_to(simulations_root).as_posix()
    info = results.get("info", {})
    return {
        "id": relative_dir,
        "name": relative_dir,
        "timestamp": results.get("timestamp"),
        "domain": info.get("environment_info", {}).get("domain_name"),
        "agent_model": info.get("agent_info", {}).get("llm"),
        "simulation_count": len(simulations),
        "pass_rate": passed / len(rewards) if rewards else None,
    }


def discover_runs(simulations_root: Path) -> list[dict]:
    """Discover valid results files below the configured simulations root."""
    runs = []
    if not simulations_root.is_dir():
        return runs
    for results_path in simulations_root.rglob("results.json"):
        try:
            runs.append(summarize_run(results_path, simulations_root))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return sorted(runs, key=lambda item: item.get("timestamp") or "", reverse=True)


def resolve_results_path(run_id: str, simulations_root: Path) -> Path:
    """Resolve a run identifier without allowing traversal outside the root."""
    root = simulations_root.resolve()
    candidate = (root / unquote(run_id) / "results.json").resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise FileNotFoundError(run_id)
    return candidate


class ViewerHandler(SimpleHTTPRequestHandler):
    """Serve static viewer assets and read-only simulation APIs."""

    simulations_root: Path

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/runs":
            self.send_json(
                {
                    "root": str(self.simulations_root),
                    "runs": discover_runs(self.simulations_root),
                }
            )
            return
        if parsed.path.startswith("/api/runs/"):
            run_id = parsed.path.removeprefix("/api/runs/")
            try:
                results_path = resolve_results_path(run_id, self.simulations_root)
                self.send_json(load_results(results_path))
            except FileNotFoundError:
                self.send_json({"error": "Run not found"}, HTTPStatus.NOT_FOUND)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        super().do_GET()

    def log_message(self, message_format: str, *args: object) -> None:
        if self.path.startswith("/api/") and args and str(args[1]) == "200":
            return
        super().log_message(message_format, *args)


def create_server(simulations_root: Path, host: str, port: int) -> ThreadingHTTPServer:
    """Create a localhost viewer server."""
    handler = type(
        "ConfiguredViewerHandler",
        (ViewerHandler,),
        {"simulations_root": simulations_root.resolve()},
    )
    return ThreadingHTTPServer((host, port), handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulations-dir",
        type=Path,
        default=DEFAULT_SIMULATIONS_DIR,
        help="tau2 simulations directory (default: data/simulations)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate discovery and exit without starting the server",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulations_root = args.simulations_dir.expanduser().resolve()
    if args.check:
        runs = discover_runs(simulations_root)
        print(f"Found {len(runs)} run(s) in {simulations_root}")
        return

    server = create_server(simulations_root, args.host, args.port)
    url = f"http://{args.host}:{server.server_port}/"
    print(f"tau² Trace Viewer: {url}")
    print(f"Reading simulations from: {simulations_root}")
    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
