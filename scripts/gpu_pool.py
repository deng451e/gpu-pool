#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Find spare GPUs across the testbed hosts listed in ~/.ssh/config.

Pool membership is the set of ``Host`` aliases matching ``--pattern``
(default ``gpu-*``, or ``$GPU_POOL_PATTERN``) minus those matching ``--exclude``
(default: CI servers), plus the local machine. GPU models are read live from
``nvidia-smi`` on each host, so the SSH config carries connection details only.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys

LOCAL = "local"
DEFAULT_PATTERNS = os.environ.get("GPU_POOL_PATTERN", "gpu-*")
DEFAULT_SSH_CONFIG = str(Path.home() / ".ssh" / "config")
# CI runners are not test capacity: "ci" as its own name component, e.g.
# gpu-ci-01 or lab-ci2, but not gpu-us-pacific.
DEFAULT_EXCLUDE = os.environ.get("GPU_POOL_EXCLUDE") or (
    r"(^|[^a-z0-9])ci\d*([^a-z0-9]|$)"
)
CLAIM_DIR = "/tmp/gpu-pool-claims"
# A claim covers the gap between picking a GPU and the job allocating memory
# on it; after that the GPU shows up as busy on its own.
CLAIM_MINUTES = 10
PROBE_CMD = (
    "hostname; "
    "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,"
    "power.draw,power.limit "
    "--format=csv,noheader,nounits; "
    "echo ---; "
    f"find {CLAIM_DIR} -type f -mmin -{CLAIM_MINUTES} -printf '%f\\n' 2>/dev/null"
)


@dataclass
class Gpu:
    """One GPU as reported by nvidia-smi, plus the pool's verdict on it."""

    index: int
    name: str
    used_mib: int
    total_mib: int
    util_pct: int
    claimed: bool
    spare: bool
    # Watts; 0 where the board does not report power.
    power_w: float = 0.0
    power_limit_w: float = 0.0
    # DCGM profiling counters (percent of the sample period with SMs / tensor
    # cores active); None where DCGM is unavailable. Only the live view fills them.
    sm_active_pct: Optional[float] = None
    tensor_active_pct: Optional[float] = None


@dataclass
class Host:
    """Probe result for one pool member."""

    alias: str
    hostname: str = ""
    error: str = ""
    gpus: list[Gpu] = field(default_factory=list)


def normalize(text: str) -> str:
    """Lower-case and drop non-alphanumerics so 'pro6000' matches 'RTX PRO 6000'.

    Args:
        text: A GPU model name or a user-supplied filter.

    Returns:
        The comparable form of ``text``.
    """
    return re.sub(r"[^a-z0-9]", "", text.lower())


def pool_aliases(config: Path, patterns: list[str], exclude: str) -> list[str]:
    """List the SSH aliases that belong to the pool.

    Args:
        config: Path to an OpenSSH client config file.
        patterns: Globs; a pool alias matches at least one.
        exclude: Case-insensitive regex; aliases it finds a match in are
            left out. Empty excludes nothing.

    Returns:
        Matching concrete (non-wildcard) ``Host`` aliases in file order,
        without duplicates.
    """
    aliases: list[str] = []
    if not config.exists():
        return aliases
    for line in config.read_text().splitlines():
        words = line.split()
        if len(words) < 2 or words[0].lower() != "host":
            continue
        for alias in words[1:]:
            if set("*?!") & set(alias) or alias in aliases:
                continue
            if exclude and re.search(exclude, alias, re.IGNORECASE):
                continue
            if any(fnmatch.fnmatch(alias, pattern) for pattern in patterns):
                aliases.append(alias)
    return aliases


def ssh_argv(ssh_config: str) -> list[str]:
    """Start an ssh command line that resolves aliases like a plain ``ssh``.

    Args:
        ssh_config: OpenSSH client config that defines the pool aliases.

    Returns:
        ``ssh`` plus ``-F ssh_config`` only when it is not the user's default
        config, because ``-F`` also skips the system-wide ``/etc/ssh/ssh_config``
        and its includes, where VPN and proxy tools add routing for hosts.
    """
    if Path(ssh_config).expanduser().resolve() == Path(DEFAULT_SSH_CONFIG).resolve():
        return ["ssh"]
    return ["ssh", "-F", ssh_config]


def run_on(
    alias: str, command: str, timeout: int, ssh_config: str
) -> subprocess.CompletedProcess[str]:
    """Run a shell command on a pool member.

    Args:
        alias: SSH alias, or ``local`` for this machine.
        command: Shell command line.
        timeout: Seconds before the call is abandoned.
        ssh_config: OpenSSH client config that defines ``alias``.

    Returns:
        The completed process with captured text output.

    Raises:
        subprocess.TimeoutExpired: If the host does not answer in time.
    """
    if alias == LOCAL:
        argv = ["bash", "-c", command]
    else:
        argv = ssh_argv(ssh_config) + [
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={timeout}",
            alias,
            command,
        ]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout + 10)


def watts(cell: str) -> float:
    """Parse an nvidia-smi power cell.

    Args:
        cell: A number of watts, or ``[N/A]`` on boards without power readings.

    Returns:
        The watts, or 0.0 for ``[N/A]``.
    """
    return 0.0 if cell.startswith("[") else float(cell)


def parse_probe(alias: str, lines: list[str], max_used_mib: int, max_util: int) -> Host:
    """Turn the output of ``PROBE_CMD`` into a ``Host`` with spare GPUs marked.

    Args:
        alias: SSH alias the output came from, or ``local``.
        lines: Output lines: hostname, one CSV row per GPU, ``---``, then the
            indices of claimed GPUs.
        max_used_mib: A spare GPU has at most this much memory in use.
        max_util: A spare GPU has at most this utilization percentage.

    Returns:
        The host and its GPUs.
    """
    split = lines.index("---")
    claims = set(lines[split + 1 :])
    host = Host(alias, hostname=lines[0])
    for row in lines[1:split]:
        index, name, used, total, util, power, limit = (
            cell.strip() for cell in row.split(",")
        )
        claimed = index in claims
        host.gpus.append(
            Gpu(
                index=int(index),
                name=name,
                used_mib=int(used),
                total_mib=int(total),
                util_pct=int(util),
                claimed=claimed,
                spare=(
                    not claimed and int(used) <= max_used_mib and int(util) <= max_util
                ),
                power_w=watts(power),
                power_limit_w=watts(limit),
            )
        )
    return host


def probe(
    alias: str, max_used_mib: int, max_util: int, timeout: int, ssh_config: str
) -> Host:
    """Query one host for its GPUs and mark the spare ones.

    Args:
        alias: SSH alias, or ``local`` for this machine.
        max_used_mib: A spare GPU has at most this much memory in use.
        max_util: A spare GPU has at most this utilization percentage.
        timeout: Seconds allowed for the SSH connection.
        ssh_config: OpenSSH client config that defines ``alias``.

    Returns:
        The host's GPUs, or a ``Host`` whose ``error`` says why it was skipped.
    """
    try:
        result = run_on(alias, PROBE_CMD, timeout, ssh_config)
    except subprocess.TimeoutExpired:
        return Host(alias, error="timed out")
    lines = result.stdout.splitlines()
    if "---" not in lines or len(lines) < 3:
        detail = (result.stderr.strip() or result.stdout.strip()).splitlines()
        return Host(alias, error=detail[-1] if detail else "no GPUs reported")
    return parse_probe(alias, lines, max_used_mib, max_util)


def probe_pool(args: argparse.Namespace) -> list[Host]:
    """Probe the local machine and every pool alias in parallel.

    Args:
        args: Parsed command-line options.

    Returns:
        One ``Host`` per distinct machine: ``local`` first, and a later alias
        that reaches an already-listed machine is dropped.
    """
    aliases = [LOCAL] + pool_aliases(
        Path(args.ssh_config), args.pattern.split(","), args.exclude
    )
    with ThreadPoolExecutor(max_workers=len(aliases)) as pool:
        hosts = list(
            pool.map(
                lambda alias: probe(
                    alias,
                    args.max_used_mib,
                    args.max_util,
                    args.timeout,
                    args.ssh_config,
                ),
                aliases,
            )
        )
    seen: set[str] = set()
    distinct: list[Host] = []
    for host in hosts:
        if host.hostname in seen:
            continue
        if host.hostname:
            seen.add(host.hostname)
        distinct.append(host)
    return distinct


def spare_gpus(host: Host, gpu_filter: str) -> list[Gpu]:
    """Return the host's spare GPUs whose model matches the filter.

    Args:
        host: A probed host.
        gpu_filter: Model substring such as ``h200`` or ``pro6000``; empty
            matches every model.

    Returns:
        Matching spare GPUs, emptiest first.
    """
    wanted = normalize(gpu_filter)
    matching = [gpu for gpu in host.gpus if gpu.spare and wanted in normalize(gpu.name)]
    return sorted(matching, key=lambda gpu: (gpu.used_mib, gpu.index))


def print_table(hosts: list[Host]) -> None:
    """Print one line per GPU for every probed host.

    Args:
        hosts: Probe results.
    """
    for host in hosts:
        if host.error:
            print(f"{host.alias}: UNREACHABLE ({host.error})")
            continue
        print(f"{host.alias} ({host.hostname})")
        for gpu in host.gpus:
            state = "SPARE" if gpu.spare else "claimed" if gpu.claimed else "busy"
            print(
                f"  {gpu.index:>2}  {gpu.name:<28} "
                f"{gpu.used_mib:>7}/{gpu.total_mib} MiB  {gpu.util_pct:>3}%  {state}"
            )


def pick(hosts: list[Host], args: argparse.Namespace) -> int:
    """Choose a host with enough spare GPUs and print how to use them.

    The local machine wins when it qualifies (no SSH hop); otherwise the host
    with the most spare matching GPUs is used.

    Args:
        hosts: Probe results.
        args: Parsed command-line options.

    Returns:
        Process exit code: 0 on success, 1 when no host qualifies.
    """
    candidates = [
        (host, spare_gpus(host, args.gpu))
        for host in hosts
        if len(spare_gpus(host, args.gpu)) >= args.count
    ]
    if not candidates:
        print(
            f"No host has {args.count} spare GPU(s) matching '{args.gpu or 'any'}'.",
            file=sys.stderr,
        )
        return 1
    host, gpus = max(
        candidates, key=lambda item: (item[0].alias == LOCAL, len(item[1]))
    )
    indices = [str(gpu.index) for gpu in gpus[: args.count]]
    if args.claim:
        paths = " ".join(f"{CLAIM_DIR}/{index}" for index in indices)
        run_on(
            host.alias,
            f"mkdir -p -m 1777 {CLAIM_DIR} && touch {paths}",
            args.timeout,
            args.ssh_config,
        )
    print(f"HOST={host.alias}")
    print(f"HOSTNAME={host.hostname}")
    print(f"GPU_MODEL={gpus[0].name}")
    print(f"CUDA_VISIBLE_DEVICES={','.join(indices)}")
    return 0


def main() -> int:
    """Parse arguments, probe the pool, and list or pick GPUs.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="", help="model filter, e.g. h200, pro6000")
    parser.add_argument("--count", type=int, default=0, help="pick this many GPUs")
    parser.add_argument(
        "--claim",
        action="store_true",
        help=f"reserve the picked GPUs for {CLAIM_MINUTES} minutes",
    )
    parser.add_argument("--max-used-mib", type=int, default=1024)
    parser.add_argument("--max-util", type=int, default=10)
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERNS,
        help="comma-separated alias globs that make up the pool",
    )
    parser.add_argument(
        "--exclude",
        default=DEFAULT_EXCLUDE,
        help="regex of aliases to leave out (default: CI servers)",
    )
    parser.add_argument("--ssh-config", default=DEFAULT_SSH_CONFIG)
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="dump probe results")
    args = parser.parse_args()

    hosts = probe_pool(args)
    if args.count:
        return pick(hosts, args)
    if args.json:
        print(json.dumps([asdict(host) for host in hosts], indent=2))
    else:
        print_table(hosts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
