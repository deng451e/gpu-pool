#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Serve a live web dashboard of GPU, CPU, memory and storage across the pool.

Runs the same per-host streaming followers as ``pool_watch.py`` and exposes
their latest reports over HTTP: ``/`` is a page that polls ``/api/state`` every
second. The page's Servers panel lists the hosts of an SSH config
(``/api/hosts``) and changes which ones are tracked (``/api/track``) without a
restart; the choice is saved in ``~/.gpu-pool/selection.json``.

There is no authentication, so it binds to 127.0.0.1 unless ``--host`` says
otherwise; reach it from another machine with an SSH port forward.

Standalone: needs only Python 3.9+ and ``ssh`` with key access to the hosts,
so it also runs on a laptop. Example::

    python3 pool_serve.py --hosts 'gpu-*,lab-node-02'
"""

# Standard
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlsplit
import argparse
import fnmatch
import json
import shutil
import sys
import threading
import time
import webbrowser

# Third Party
from gpu_pool import (
    DEFAULT_EXCLUDE,
    DEFAULT_PATTERNS,
    DEFAULT_SSH_CONFIG,
    LOCAL,
    pool_aliases,
)
from pool_watch import STALE_INTERVALS, Report, follow

PAGE = Path(__file__).with_name("pool_live.html")
STATE_DIR = Path.home() / ".gpu-pool"
# Aliases or globs, one per line, left off the default selection (but still in
# the pool for gpu_pool.py). Lives outside scripts/ so reinstalling keeps it.
HIDE_FILE = STATE_DIR / "hide"
# The Servers panel's choice: {"ssh_config": path, "hosts": [alias, ...]}.
SELECTION_FILE = STATE_DIR / "selection.json"
MAX_BODY_BYTES = 64 * 1024


def snapshot(reports: dict[str, Report], aliases: list[str], interval: float) -> dict:
    """Build the JSON document the live page draws.

    Args:
        reports: Latest report per alias; aliases without one are connecting.
        aliases: Display order, ``local`` first.
        interval: Seconds between reports, used to flag stale hosts.

    Returns:
        ``{"at": epoch seconds, "interval": seconds, "hosts": [...]}`` with one
        entry per distinct machine. A host entry has ``alias`` and ``state``
        (``ok``, ``stale``, ``connecting`` or ``unreachable``); reachable hosts
        add ``hostname``, ``age``, ``gpus``, ``disks``, ``stats``, ``nodes``
        (per-core CPU percent by NUMA node) and ``membw_gbps`` (memory traffic
        per socket, empty when the host does not expose it), and
        unreachable ones add ``error``.
    """
    now = time.time()
    hosts: list[dict] = []
    seen: set[str] = set()
    for alias in aliases:
        report = reports.get(alias)
        if report is None:
            hosts.append({"alias": alias, "state": "connecting"})
            continue
        if report.host.error:
            hosts.append(
                {"alias": alias, "state": "unreachable", "error": report.host.error}
            )
            continue
        if report.host.hostname in seen:
            continue
        seen.add(report.host.hostname)
        age = now - report.at
        hosts.append(
            {
                "alias": alias,
                "state": "stale" if age > STALE_INTERVALS * interval else "ok",
                "hostname": report.host.hostname,
                "age": round(age, 1),
                "gpus": [asdict(gpu) for gpu in report.host.gpus],
                "disks": [asdict(disk) for disk in report.disks],
                "stats": report.stats,
                "nodes": report.nodes,
                "membw_gbps": report.membw_gbps,
            }
        )
    return {"at": now, "interval": interval, "hosts": hosts}


def config_hosts(config: Path) -> list[dict]:
    """List the concrete hosts an SSH config defines.

    Args:
        config: Path to an OpenSSH client config. ``Include`` is not followed.

    Returns:
        ``{"alias", "hostname"}`` per non-wildcard ``Host`` alias in file
        order; ``hostname`` is the ``HostName`` of its block, or empty.
    """
    hosts: dict[str, str] = {}
    block: list[str] = []
    for line in config.read_text().splitlines():
        words = line.split()
        if len(words) < 2 or words[0].startswith("#"):
            continue
        key = words[0].lower()
        if key == "host":
            block = [a for a in words[1:] if not set("*?!") & set(a)]
            for alias in block:
                hosts.setdefault(alias, "")
        elif key == "match":
            block = []
        elif key == "hostname":
            for alias in block:
                hosts[alias] = hosts[alias] or words[1]
    return [{"alias": alias, "hostname": name} for alias, name in hosts.items()]


def matches_any(alias: str, globs: list[str]) -> bool:
    """Tell whether an alias matches any of the globs.

    Args:
        alias: SSH alias.
        globs: Shell-style patterns.

    Returns:
        True when at least one glob matches.
    """
    return any(fnmatch.fnmatch(alias, glob) for glob in globs)


def hidden_patterns() -> list[str]:
    """Read the dashboard's hidden-host list.

    Returns:
        Globs from ``HIDE_FILE``, skipping blank lines and ``#`` comments;
        empty when the file does not exist.
    """
    if not HIDE_FILE.exists():
        return []
    lines = (line.strip() for line in HIDE_FILE.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def load_selection() -> Optional[dict]:
    """Read the saved Servers-panel choice.

    Returns:
        ``{"ssh_config": str, "hosts": [str, ...]}``, or None when nothing
        has been saved yet.
    """
    if not SELECTION_FILE.exists():
        return None
    return json.loads(SELECTION_FILE.read_text())


def default_aliases(args: argparse.Namespace, config: Path) -> list[str]:
    """Choose the hosts to track when neither ``--hosts`` nor a saved choice does.

    Args:
        args: Parsed options: ``pattern`` and ``exclude`` select pool hosts,
            ``hide`` (comma-separated globs) drops some, ``no_local`` skips
            this machine.
        config: SSH config to read aliases from.

    Returns:
        Aliases in display order; ``local`` first when this machine has
        ``nvidia-smi``, so a laptop launch shows only the remote servers.
    """
    local = [LOCAL] if shutil.which("nvidia-smi") and not args.no_local else []
    hide = [glob for glob in args.hide.split(",") if glob]
    return local + [
        alias
        for alias in pool_aliases(config, args.pattern.split(","), args.exclude)
        if not matches_any(alias, hide)
    ]


def initial_selection(args: argparse.Namespace) -> tuple[str, list[str]]:
    """Decide the SSH config and hosts to start with.

    Precedence: ``--hosts`` (globs, not saved), then the saved selection, then
    the pool defaults. ``--ssh-config`` overrides the saved config path.

    Args:
        args: Parsed options.

    Returns:
        The SSH config path and the aliases to track.
    """
    saved = load_selection()
    config = args.ssh_config or (saved or {}).get("ssh_config") or DEFAULT_SSH_CONFIG
    path = Path(config).expanduser()
    if args.hosts:
        wanted = args.hosts.split(",")
        local = [LOCAL] if LOCAL in wanted else []
        return config, local + pool_aliases(path, wanted, "")
    if saved is not None and str(Path(saved["ssh_config"]).expanduser()) == str(path):
        return config, list(saved["hosts"])
    return config, default_aliases(args, path)


class Tracker:
    """Owns one follower thread per tracked host and changes the set live."""

    def __init__(self, args: argparse.Namespace, reports: dict[str, Report]) -> None:
        """Create an empty tracker.

        Args:
            args: Parsed options passed to every follower.
            reports: Shared report map the followers update.
        """
        self.args = args
        self.reports = reports
        self.lock = threading.Lock()
        self.ssh_config = ""
        self.order: list[str] = []
        self.stops: dict[str, threading.Event] = {}

    def track(self, ssh_config: str, aliases: list[str]) -> None:
        """Follow exactly ``aliases`` from ``ssh_config``.

        Hosts no longer listed are disconnected; a changed config restarts
        every follower so they resolve aliases from the new file.

        Args:
            ssh_config: SSH config path the aliases come from.
            aliases: Hosts to follow, in display order.
        """
        with self.lock:
            if ssh_config != self.ssh_config:
                drop = list(self.stops)
            else:
                drop = [alias for alias in self.stops if alias not in aliases]
            for alias in drop:
                self.stops.pop(alias).set()
                self.reports.pop(alias, None)
            options = argparse.Namespace(
                **{**vars(self.args), "ssh_config": ssh_config}
            )
            for alias in aliases:
                if alias in self.stops:
                    continue
                stop = threading.Event()
                self.stops[alias] = stop
                threading.Thread(
                    target=follow,
                    args=(alias, options, self.reports, stop),
                    daemon=True,
                ).start()
            self.ssh_config = ssh_config
            self.order = list(aliases)

    def current(self) -> tuple[str, list[str]]:
        """Return the SSH config and the tracked aliases in display order."""
        with self.lock:
            return self.ssh_config, list(self.order)


def hosts_document(tracker: Tracker, config: Optional[str]) -> dict:
    """Describe the hosts a config offers, for the Servers panel.

    Args:
        tracker: The running tracker.
        config: Config path to list; None lists the tracked one.

    Returns:
        ``{"ssh_config", "hosts": [{"alias", "hostname", "tracked"}],
        "local_gpus"}``, plus ``error`` when the file cannot be read.
    """
    tracked_config, tracked = tracker.current()
    config = config or tracked_config
    path = Path(config).expanduser()
    document: dict = {
        "ssh_config": config,
        "local_gpus": shutil.which("nvidia-smi") is not None,
        "hosts": [],
    }
    try:
        entries = config_hosts(path)
    except OSError as error:
        document["error"] = f"cannot read {path}: {error.strerror}"
        return document
    if document["local_gpus"]:
        entries.insert(0, {"alias": LOCAL, "hostname": "this machine"})
    same = str(path) == str(Path(tracked_config).expanduser())
    for entry in entries:
        entry["tracked"] = same and entry["alias"] in tracked
    document["hosts"] = entries
    return document


def make_handler(tracker: Tracker, interval: float, host: str, port: int) -> type:
    """Create the request handler bound to the tracker.

    Args:
        tracker: Owns the followers and the tracked host list.
        interval: Seconds between reports.
        host: Address the server binds, also accepted in Host/Origin.
        port: Port the server listens on, for the Host/Origin check.

    Returns:
        A handler class serving ``/``, ``/api/state``, ``/api/hosts`` and
        ``POST /api/track``.
    """
    allowed = {f"127.0.0.1:{port}", f"localhost:{port}", f"{host}:{port}"}

    class Handler(BaseHTTPRequestHandler):
        """Serves the live page, its state, and the host selection API."""

        def do_GET(self) -> None:
            """Answer the page, the state document or the host list."""
            url = urlsplit(self.path)
            if url.path == "/":
                self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/state" and self.trusted():
                _, aliases = tracker.current()
                state = snapshot(tracker.reports, aliases, interval)
                self.reply_json(200, state)
            elif url.path == "/api/hosts" and self.trusted():
                config = parse_qs(url.query).get("config", [None])[0]
                self.reply_json(200, hosts_document(tracker, config))
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            """Apply and save a new selection from the Servers panel."""
            if urlsplit(self.path).path != "/api/track" or not self.trusted():
                self.send_error(404)
                return
            # Requiring JSON makes a cross-site form unable to post here.
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.send_error(415)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_BODY_BYTES:
                self.send_error(413)
                return
            try:
                body = json.loads(self.rfile.read(length))
                config, hosts = str(body["ssh_config"]), list(body["hosts"])
            except (ValueError, KeyError, TypeError):
                self.reply_json(400, {"error": "expected {ssh_config, hosts}"})
                return
            document = hosts_document(tracker, config)
            if "error" in document:
                self.reply_json(400, document)
                return
            known = [entry["alias"] for entry in document["hosts"]]
            unknown = [alias for alias in hosts if alias not in known]
            if unknown:
                message = f"not in {config}: {', '.join(unknown)}"
                self.reply_json(400, {"error": message})
                return
            ordered = [alias for alias in known if alias in hosts]
            tracker.track(config, ordered)
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            selection = {"ssh_config": config, "hosts": ordered}
            SELECTION_FILE.write_text(json.dumps(selection, indent=2) + "\n")
            self.reply_json(200, hosts_document(tracker, None))

        def trusted(self) -> bool:
            """Refuse requests a web page on another origin could make.

            Checks Host (DNS rebinding) and, when present, Origin.

            Returns:
                True for requests addressed to this server from its own page.
            """
            origin = self.headers.get("Origin")
            if self.headers.get("Host") not in allowed or (
                origin is not None
                and origin not in {f"http://{host}" for host in allowed}
            ):
                self.send_error(403)
                return False
            return True

        def reply_json(self, status: int, document: dict) -> None:
            """Send a JSON document."""
            self.reply(status, json.dumps(document).encode(), "application/json")

        def reply(self, status: int, body: bytes, kind: str) -> None:
            """Send a complete, uncached response."""
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Stay quiet: one request per viewer per second is not worth logging."""

    return Handler


def main() -> int:
    """Parse arguments, start one follower per host and serve until stopped.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="address to bind")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=1.0, help="seconds")
    parser.add_argument("--max-used-mib", type=int, default=1024)
    parser.add_argument("--max-util", type=int, default=10)
    parser.add_argument(
        "--hosts",
        default="",
        help="comma-separated SSH aliases or globs to track, e.g. "
        "'gpu-node-16,lab-node-0*'; 'local' adds this machine. Overrides the "
        "saved selection for this run",
    )
    parser.add_argument(
        "--hide",
        default=",".join(hidden_patterns()),
        help="comma-separated aliases or globs left out of the default "
        f"selection (default: {HIDE_FILE})",
    )
    parser.add_argument("--pattern", default=DEFAULT_PATTERNS)
    parser.add_argument("--exclude", default=DEFAULT_EXCLUDE)
    parser.add_argument("--no-local", action="store_true", help="skip this machine")
    parser.add_argument(
        "--ssh-config",
        default="",
        help=f"SSH config to read hosts from (default: the saved one, else "
        f"{DEFAULT_SSH_CONFIG})",
    )
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument(
        "--list", action="store_true", help="print the selected hosts and exit"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open the page"
    )
    args = parser.parse_args()

    config, aliases = initial_selection(args)
    if args.list:
        print("\n".join(aliases))
        return 0
    reports: dict[str, Report] = {}
    tracker = Tracker(args, reports)
    tracker.track(config, aliases)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(tracker, args.interval, args.host, args.port),
    )
    url = f"http://127.0.0.1:{args.port}/"
    tracked = ", ".join(aliases) or "none yet: pick them in the Servers panel"
    print(f"GPU pool live view: {url}  ({tracked})", flush=True)
    print("Ctrl-C to stop.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
