#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Live terminal view of GPU, CPU, memory and storage usage across the pool.

Keeps one SSH connection open per pool host; each runs a small loop that
reports ``nvidia-smi``, CPU, memory and ``df`` every ``--interval`` seconds, so a
refresh costs no new connections. Pool membership and the definition of a spare GPU
come from ``gpu_pool.py``. Stop with Ctrl-C.
"""

# Standard
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import argparse
import shutil
import subprocess
import sys
import threading
import time

# Third Party
from gpu_pool import (
    DEFAULT_EXCLUDE,
    DEFAULT_PATTERNS,
    DEFAULT_SSH_CONFIG,
    LOCAL,
    PROBE_CMD,
    Host,
    parse_probe,
    pool_aliases,
    ssh_argv,
)
from pool_report import HOST_STAT_CMD, host_stats

BEGIN, STAT, DCGM, DISK, END = "@@BEGIN", "@@STAT", "@@DCGM", "@@DISK", "@@END"
DF_CMD = (
    "df -B1 --output=target,size,avail "
    "-x tmpfs -x devtmpfs -x overlay -x squashfs -x efivarfs -x vfat | tail -n +2"
)
# After the host stat line: one "n <node> <cpulist>" per NUMA node, one
# "m <bytes>" per L3 domain when resctrl memory-bandwidth monitoring is mounted
# (it needs root to mount, so often absent), and one "c <cpu> <busy> <total>"
# per logical CPU in jiffies since boot.
TOPOLOGY_CMD = (
    "for n in /sys/devices/system/node/node[0-9]*; do "
    'echo "n ${n##*node} $(cat $n/cpulist)"; done; '
    "for f in /sys/fs/resctrl/mon_data/mon_L3_*/mbm_total_bytes; do "
    '[ -r "$f" ] && echo "m $(cat $f)"; done; '
    'awk \'/^cpu[0-9]/{print "c", substr($1,4), $2+$3+$4+$7+$8+$9, '
    "$2+$3+$4+$5+$6+$7+$8+$9}' /proc/stat"
)
# DCGM profiling fields: 1002 = SM active, 1004 = tensor-core active, each the
# fraction of the sample period. ``dcgmi dmon`` runs in the background for the
# life of the connection, appending one row per GPU per sample to a temp file;
# each report prints the latest "g <gpu> <sm> <tensor>" per GPU and empties the
# file. Hosts without a running DCGM host engine print nothing.
DCGM_START = (
    "dcgm=$(mktemp); "
    "dcgmi dmon -e 1002,1004 -d {ms} >> $dcgm 2>/dev/null & dmon=$!; "
    "trap 'kill $dmon 2>/dev/null; rm -f $dcgm' EXIT; "
)
DCGM_READ = (
    'awk \'$1=="GPU" && $3!="N/A" {s[$2]=$3; t[$2]=$4} '
    'END {for (g in s) print "g", g, s[g], t[g]}\' $dcgm; : > $dcgm'
)
RECONNECT_SECONDS = 5
# A host whose last report is older than this many intervals is shown as stale.
STALE_INTERVALS = 5
DISK_WARN_PCT = 80
DISK_FULL_PCT = 90
GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m",
    "\033[33m",
    "\033[31m",
    "\033[2m",
    "\033[1m",
    "\033[0m",
)


@dataclass
class Disk:
    """One filesystem as reported by df."""

    mount: str
    free_gb: float
    used_pct: float


@dataclass
class Report:
    """Latest state of one pool host."""

    host: Host
    disks: list[Disk] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    at: float = 0.0
    nodes: list[dict] = field(default_factory=list)
    membw_gbps: list[float] = field(default_factory=list)


@dataclass
class Counters:
    """Raw counters from one sample, kept to diff against the next one."""

    at: float = 0.0
    cores: dict[int, tuple[int, int]] = field(default_factory=dict)
    membw_bytes: list[int] = field(default_factory=list)


def expand_cpulist(text: str) -> list[int]:
    """Expand a kernel CPU list such as ``0-3,8,10-11`` into CPU numbers.

    Args:
        text: The list as sysfs prints it.

    Returns:
        The CPU numbers in the order given.
    """
    cpus: list[int] = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        cpus.extend(range(int(first), int(last or first) + 1))
    return cpus


def topology_usage(
    lines: list[str], previous: Counters, now: float
) -> tuple[list[dict], list[float], Counters]:
    """Turn one ``TOPOLOGY_CMD`` sample into per-core and memory-bandwidth rates.

    Args:
        lines: The sample's ``n``, ``m`` and ``c`` lines.
        previous: Counters from the sample before; empty on the first one.
        now: Time this sample was read, in epoch seconds.

    Returns:
        ``nodes``: one ``{"id": node, "cpus": [cpu, ...], "cores": [busy percent,
        ...]}`` per NUMA node, both in CPU-number order; a core reads 0 where
        there is no earlier sample.
        ``membw_gbps``: memory traffic per L3 domain in GB/s since the previous
        sample; empty when the host does not expose it or on the first sample.
        The third item is this sample's counters, for the next call.
    """
    current = Counters(at=now)
    layout: list[tuple[int, list[int]]] = []
    for line in lines:
        kind, _, rest = line.partition(" ")
        if kind == "n":
            node, cpulist = rest.split()
            layout.append((int(node), expand_cpulist(cpulist)))
        elif kind == "m" and rest.isdigit():
            current.membw_bytes.append(int(rest))
        elif kind == "c":
            cpu, busy, total = (int(cell) for cell in rest.split())
            current.cores[cpu] = (busy, total)
    nodes = []
    for node, cpus in sorted(layout):
        cores = []
        for cpu in sorted(cpus):
            busy, total = current.cores[cpu]
            busy0, total0 = previous.cores.get(cpu, (busy, total))
            cores.append(round(100 * (busy - busy0) / max(total - total0, 1), 1))
        nodes.append({"id": node, "cpus": sorted(cpus), "cores": cores})
    membw: list[float] = []
    if len(previous.membw_bytes) == len(current.membw_bytes) and now > previous.at:
        membw = [
            round((new - previous.membw_bytes[i]) / (now - previous.at) / 1e9, 2)
            for i, new in enumerate(current.membw_bytes)
        ]
    return nodes, membw, current


def follow(
    alias: str,
    args: argparse.Namespace,
    reports: dict[str, Report],
    stop: Optional[threading.Event] = None,
) -> None:
    """Keep ``reports[alias]`` current from a streaming connection to the host.

    Reconnects after ``RECONNECT_SECONDS`` when the connection ends, recording
    the reason in the report meanwhile. Runs until ``stop`` is set (closing the
    connection within one report interval) or the process exits.

    Args:
        alias: SSH alias, or ``local`` for this machine.
        args: Parsed command-line options.
        reports: Shared map of alias to latest report, updated in place.
        stop: Set to end following this host; None follows forever.
    """
    stop = stop or threading.Event()
    loop = (
        DCGM_START.format(ms=max(100, int(args.interval * 1000)))
        + f"while :; do echo {BEGIN}; {PROBE_CMD}; echo {STAT}; {HOST_STAT_CMD}; "
        f"{TOPOLOGY_CMD}; "
        f"echo {DCGM}; {DCGM_READ}; "
        f"echo {DISK}; {DF_CMD}; "
        f"echo {END}; sleep {args.interval}; done"
    )
    if alias == LOCAL:
        argv = ["bash"]
    else:
        argv = ssh_argv(args.ssh_config) + ["-o", "BatchMode=yes"]
        argv += ["-o", f"ConnectTimeout={args.timeout}"]
        argv += ["-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
        # The loop goes in on stdin so it runs under bash whatever the login shell.
        argv += [alias, "bash"]
    while not stop.is_set():
        stream = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stream.stdin.write(loop + "\n")
        stream.stdin.close()
        block: list[str] = []
        last_stat = ""
        counters = Counters()
        # Latest DCGM reading per GPU, kept when a report has no new sample.
        activity: dict[int, tuple[float, float]] = {}
        for raw in stream.stdout:
            if stop.is_set():
                break
            line = raw.rstrip("\n")
            if line == BEGIN:
                block = []
            elif line == END and {STAT, DCGM, DISK} <= set(block):
                stat_at, dcgm_at = block.index(STAT), block.index(DCGM)
                cut = block.index(DISK)
                stat = block[stat_at + 1]
                # The first sample has nothing to diff against: CPU reads 0.
                stats = host_stats(last_stat or stat, stat)
                last_stat = stat
                now = time.time()
                nodes, membw, counters = topology_usage(
                    block[stat_at + 2 : dcgm_at], counters, now
                )
                for row in block[dcgm_at + 1 : cut]:
                    _, gpu, sm, tensor = row.split()
                    activity[int(gpu)] = (100 * float(sm), 100 * float(tensor))
                disks = []
                for row in block[cut + 1 :]:
                    mount, size, avail = row.split()
                    disks.append(
                        Disk(
                            mount,
                            int(avail) / 1024**3,
                            100 * (1 - int(avail) / int(size)),
                        )
                    )
                host = parse_probe(
                    alias, block[:stat_at], args.max_used_mib, args.max_util
                )
                for gpu in host.gpus:
                    if gpu.index in activity:
                        gpu.sm_active_pct, gpu.tensor_active_pct = activity[gpu.index]
                reports[alias] = Report(host, disks, stats, now, nodes, membw)
            else:
                block.append(line)
        if stop.is_set():
            # Closing ssh ends the remote loop, whose EXIT trap stops dcgmi.
            stream.terminate()
            stream.wait()
            return
        reason = stream.stderr.read().strip().splitlines()
        stream.wait()
        reports[alias] = Report(
            Host(alias, error=reason[-1] if reason else "connection closed")
        )
        stop.wait(RECONNECT_SECONDS)


def render(reports: dict[str, Report], aliases: list[str], interval: float) -> str:
    """Draw every host's GPUs and filesystems as one text frame.

    Args:
        reports: Latest report per alias; aliases without one are connecting.
        aliases: Display order, ``local`` first.
        interval: Seconds between reports, used to flag stale hosts.

    Returns:
        The frame, ANSI-coloured, without a trailing newline.
    """
    width = shutil.get_terminal_size((100, 40)).columns
    now = time.time()
    body: list[str] = []
    spare = total = 0
    seen: set[str] = set()
    for alias in aliases:
        report = reports.get(alias)
        if report is None:
            body += [f"{BOLD}{alias}{RESET}  {DIM}connecting...{RESET}", ""]
            continue
        host = report.host
        if host.error:
            body += [f"{BOLD}{alias}{RESET}  {RED}unreachable: {host.error}{RESET}", ""]
            continue
        if host.hostname in seen:
            continue
        seen.add(host.hostname)
        host_spare = sum(gpu.spare for gpu in host.gpus)
        spare += host_spare
        total += len(host.gpus)
        models = ", ".join(
            sorted({gpu.name.removeprefix("NVIDIA ") for gpu in host.gpus})
        )
        stale = now - report.at > STALE_INTERVALS * interval
        note = f"  {YELLOW}stale {now - report.at:.0f}s{RESET}" if stale else ""
        stats = report.stats
        body.append(
            f"{BOLD}{host.hostname}{RESET}  {models}  "
            f"{GREEN}{host_spare}{RESET}/{len(host.gpus)} spare  "
            f"{DIM}cpu{RESET} {stats['cpu_pct']:.0f}% of {stats['cpu_cores']} cores  "
            f"{DIM}ram{RESET} {stats['mem_used_gb']:.0f}/{stats['mem_total_gb']:.0f}G "
            f"({stats['mem_pct']:.0f}%){note}"
        )
        rows = {"GPU ": "", "mem ": "", "util": ""}
        for gpu in host.gpus:
            tint = GREEN if gpu.spare else YELLOW if gpu.claimed else ""
            rows["GPU "] += f"{tint}{gpu.index:>6}{RESET}"
            rows["mem "] += f"{tint}{gpu.used_mib / 1024:>5.0f}G{RESET}"
            rows["util"] += f"{tint}{gpu.util_pct:>5}%{RESET}"
        body += [f"  {DIM}{label}{RESET}{cells}" for label, cells in rows.items()]
        line, used = f"  {DIM}disk{RESET}", 6
        for disk in sorted(report.disks, key=lambda item: item.mount):
            tint = (
                RED
                if disk.used_pct >= DISK_FULL_PCT
                else YELLOW
                if disk.used_pct >= DISK_WARN_PCT
                else ""
            )
            text = f"{disk.mount} {disk.free_gb:.0f}G free ({disk.used_pct:.0f}%)"
            if used + len(text) + 3 > width and used > 6:
                body.append(line)
                line, used = "      ", 6
            line += f"   {tint}{text}{RESET}"
            used += len(text) + 3
        body += [line, ""]
    head = (
        f"{BOLD}GPU testbed pool{RESET}  {time.strftime('%H:%M:%S')}  "
        f"{GREEN}{spare}{RESET}/{total} GPUs spare  "
        f"{DIM}green = spare, yellow = claimed; disk yellow >= {DISK_WARN_PCT}%, "
        f"red >= {DISK_FULL_PCT}%{RESET}"
    )
    return "\n".join([head, ""] + body).rstrip("\n")


def main() -> int:
    """Parse arguments, start one follower per host and redraw until stopped.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=1.0, help="seconds")
    parser.add_argument(
        "--once", action="store_true", help="print one frame after 5 s and exit"
    )
    parser.add_argument("--max-used-mib", type=int, default=1024)
    parser.add_argument("--max-util", type=int, default=10)
    parser.add_argument("--pattern", default=DEFAULT_PATTERNS)
    parser.add_argument("--exclude", default=DEFAULT_EXCLUDE)
    parser.add_argument("--ssh-config", default=DEFAULT_SSH_CONFIG)
    parser.add_argument("--timeout", type=int, default=8)
    args = parser.parse_args()

    aliases = [LOCAL] + pool_aliases(
        Path(args.ssh_config), args.pattern.split(","), args.exclude
    )
    reports: dict[str, Report] = {}
    for alias in aliases:
        threading.Thread(
            target=follow, args=(alias, args, reports), daemon=True
        ).start()
    if args.once:
        time.sleep(5)
        print(render(reports, aliases, args.interval))
        return 0
    sys.stdout.write("\033[?1049h\033[?25l")
    try:
        while True:
            frame = render(reports, aliases, args.interval)
            sys.stdout.write(
                "\033[H" + frame.replace("\n", "\033[K\n") + "\033[K\033[J"
            )
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    finally:
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
