#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Snapshot GPU, RDMA and storage state of the testbed pool as JSON row files.

Writes ``hosts.json``, ``gpus.json``, ``storage.json``, ``rdma_devices.json``
and ``rdma_links.json`` into ``--out``; each is a JSON array of flat row
objects, the shape the pool dashboard's datasets read. Pool membership and the
definition of a spare GPU come from ``gpu_pool.py``.

RDMA links are checked with a 1000-message ``ibv_rc_pingpong`` between the
same-named network interface on every pair of hosts (device names can differ
between hosts; interface names such as ``bond0`` or ``rail1`` line up). That
proves the RDMA path works and gives a round-trip time; it is not a throughput
benchmark.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys
import time

# Third Party
from gpu_pool import (
    DEFAULT_EXCLUDE,
    DEFAULT_PATTERNS,
    DEFAULT_SSH_CONFIG,
    LOCAL,
    Host,
    pool_aliases,
    probe,
    run_on,
    ssh_argv,
)

PINGPONG_BASE_PORT = 18600
# Per RDMA device: name, netdev, rate, state, link layer, NUMA node, and the
# index of the first routable RoCE v2 GID (link-local GIDs do not cross hosts).
RDMA_CMD = r"""
for d in /sys/class/infiniband/*; do
  [ -d "$d" ] || continue
  n=$(basename $d); gid=-1
  for i in $(seq 0 15); do
    t=$(cat $d/ports/1/gid_attrs/types/$i 2>/dev/null)
    g=$(cat $d/ports/1/gids/$i 2>/dev/null)
    case "$t|$g" in
      "RoCE v2|fe80"*|"RoCE v2|0000:0000:0000:0000:0000:0000:0000:0000") ;;
      "RoCE v2|"*) gid=$i; break ;;
    esac
  done
  p=$d/ports/1
  echo "$n|$(cat $p/gid_attrs/ndevs/0 2>/dev/null)|$(cat $p/rate)|$(cat $p/state)" \
    "|$(cat $p/link_layer)|$(cat $d/device/numa_node)|$gid"
done
"""
# One line: busy and total CPU jiffies since boot, core count, and total and
# available memory in kB. Two samples give CPU usage over the gap between them.
HOST_STAT_CMD = (
    "echo $(awk 'NR==1{print $2+$3+$4+$7+$8+$9, $2+$3+$4+$5+$6+$7+$8+$9}' /proc/stat)"
    " $(nproc)"
    " $(awk '/^MemTotal/{t=$2} /^MemAvailable/{a=$2} END{print t, a}' /proc/meminfo)"
)
DETAIL_CMD = (
    "ip -4 route get 1.1.1.1 | sed -n 's/.* src \\([0-9.]*\\).*/\\1/p'; "
    + HOST_STAT_CMD
    + "; sleep 0.5; "
    + HOST_STAT_CMD
    + "; echo ---; "
    + RDMA_CMD
    + "echo ---; "
    + "df -B1 --output=target,fstype,size,used,avail "
    + "-x tmpfs -x devtmpfs -x overlay -x squashfs -x efivarfs -x vfat"
)
GIB = 1024**3


def host_stats(first: str, second: str) -> dict:
    """Derive CPU and memory usage from two ``HOST_STAT_CMD`` samples.

    Args:
        first: The earlier sample line.
        second: The later sample line.

    Returns:
        ``cpu_pct`` (busy share of all cores between the samples),
        ``cpu_cores``, ``mem_used_gb``, ``mem_total_gb`` and ``mem_pct``, where
        used memory is total minus available.
    """
    busy0, total0 = (int(cell) for cell in first.split()[:2])
    busy1, total1, cores, mem_total, mem_avail = (int(c) for c in second.split())
    return {
        "cpu_pct": round(100 * (busy1 - busy0) / max(total1 - total0, 1), 1),
        "cpu_cores": cores,
        "mem_used_gb": round((mem_total - mem_avail) / 1024**2, 1),
        "mem_total_gb": round(mem_total / 1024**2, 1),
        "mem_pct": round(100 * (mem_total - mem_avail) / mem_total, 1),
    }


def host_detail(alias: str, timeout: int, ssh_config: str) -> dict:
    """Collect one host's address, RDMA devices and filesystems.

    Args:
        alias: SSH alias, or ``local`` for this machine.
        timeout: Seconds allowed for the SSH connection.
        ssh_config: OpenSSH client config that defines ``alias``.

    Returns:
        ``{"ip": str, "stats": dict, "rdma": [row, ...], "storage": [row, ...]}``
        with rows not yet tagged with the host name.

    Raises:
        subprocess.TimeoutExpired: If the host does not answer in time.
    """
    out = run_on(alias, DETAIL_CMD, timeout, ssh_config).stdout
    ip_part, rdma_part, df_part = out.split("---\n")
    rdma = []
    for line in rdma_part.splitlines():
        device, netdev, rate, state, layer, numa, gid = (
            cell.strip() for cell in line.split("|")
        )
        rdma.append(
            {
                "device": device,
                "netdev": netdev,
                "rate_gbps": float(rate.split()[0]),
                "state": state.split(":")[-1].strip(),
                "link_layer": layer,
                "numa": int(numa),
                "gid_index": int(gid),
            }
        )
    storage = []
    for line in df_part.splitlines()[1:]:
        mount, fstype, size, used, avail = line.split()
        storage.append(
            {
                "mount": mount,
                "fstype": fstype,
                "size_gb": round(int(size) / GIB, 1),
                "used_gb": round(int(used) / GIB, 1),
                "free_gb": round(int(avail) / GIB, 1),
                "used_pct": round(100 * (1 - int(avail) / int(size)), 1),
            }
        )
    ip, first, second = ip_part.splitlines()
    return {
        "ip": ip,
        "stats": host_stats(first, second),
        "rdma": rdma,
        "storage": storage,
    }


def pingpong(
    server: dict, client: dict, netdev: str, port: int, timeout: int, ssh_config: str
) -> dict:
    """Run one RDMA ping-pong from ``client`` to ``server`` over ``netdev``.

    Args:
        server: Host record (``alias``, ``host``, ``ip``, ``rdma``) that listens.
        client: Host record that connects.
        netdev: Network interface name backed by an RDMA device on both hosts.
        port: TCP port for the connection setup exchange.
        timeout: Seconds allowed for the SSH connection.
        ssh_config: OpenSSH client config that defines the aliases.

    Returns:
        A link row: both host names, the interface, each side's RDMA device,
        ``ok`` and, on success, the round-trip time in microseconds; on failure
        the tool's last message.
    """
    listen_dev, listen_gid = server["rdma"][netdev]
    connect_dev, connect_gid = client["rdma"][netdev]
    tool = "timeout 10 ibv_rc_pingpong"
    listen_argv = f"{tool} -d {listen_dev} -g {listen_gid} -p {port}"
    connect = f"{tool} -d {connect_dev} -g {connect_gid} -p {port} {server['ip']}"
    if server["alias"] == LOCAL:
        argv = ["bash", "-c", listen_argv]
    else:
        argv = ssh_argv(ssh_config) + ["-o", "BatchMode=yes", server["alias"]]
        argv.append(listen_argv)
    listener = subprocess.Popen(
        argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(1.5)
    result = run_on(client["alias"], connect, timeout + 10, ssh_config)
    listener.wait()
    match = re.search(r"= ([\d.]+) usec/iter", result.stdout)
    detail = (result.stderr.strip() or result.stdout.strip()).splitlines()
    return {
        "link": f"{client['host']} > {server['host']} {netdev}",
        "from_host": client["host"],
        "to_host": server["host"],
        "netdev": netdev,
        "from_device": connect_dev,
        "to_device": listen_dev,
        "ok": match is not None,
        "rtt_usec": float(match.group(1)) if match else None,
        "detail": "" if match else (detail[-1] if detail else "no output"),
    }


def link_rows(details: list[dict], timeout: int, ssh_config: str) -> list[dict]:
    """Ping-pong every RDMA interface shared by every pair of hosts.

    Args:
        details: Host records with ``alias``, ``host``, ``ip`` and ``rdma``
            (interface name to its RDMA device and routable GID index).
        timeout: Seconds allowed for each SSH connection.
        ssh_config: OpenSSH client config that defines the aliases.

    Returns:
        One row per (pair, interface), in pair then interface order.
    """
    jobs = []
    for server, client in combinations(details, 2):
        for netdev in sorted(set(server["rdma"]) & set(client["rdma"])):
            jobs.append((server, client, netdev, PINGPONG_BASE_PORT + len(jobs)))
    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(lambda job: pingpong(*job, timeout, ssh_config), jobs))


def collect(args: argparse.Namespace) -> dict[str, list[dict]]:
    """Probe the whole pool and build every dataset.

    Args:
        args: Parsed command-line options.

    Returns:
        Dataset name to its rows.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    aliases = [LOCAL] + pool_aliases(
        Path(args.ssh_config), args.pattern.split(","), args.exclude
    )
    with ThreadPoolExecutor(max_workers=len(aliases)) as pool:
        probed: list[Host] = list(
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
    data: dict[str, list[dict]] = {
        name: [] for name in ("hosts", "gpus", "storage", "rdma_devices", "rdma_links")
    }
    details = []
    seen: set[str] = set()
    for host in probed:
        if host.hostname in seen:
            continue
        if host.error:
            data["hosts"].append(
                {
                    "host": host.alias,
                    "alias": host.alias,
                    "reachable": False,
                    "error": host.error,
                    "collected_at": stamp,
                }
            )
            continue
        seen.add(host.hostname)
        detail = host_detail(host.alias, args.timeout, args.ssh_config)
        up = [dev for dev in detail["rdma"] if dev["state"] == "ACTIVE"]
        models = sorted({gpu.name.removeprefix("NVIDIA ") for gpu in host.gpus})
        data["hosts"].append(
            {
                "host": host.hostname,
                "alias": host.alias,
                "reachable": True,
                "error": "",
                "collected_at": stamp,
                "gpu_model": ", ".join(models),
                "gpus": len(host.gpus),
                "spare_gpus": sum(gpu.spare for gpu in host.gpus),
                **detail["stats"],
                "rdma_devices": len(detail["rdma"]),
                "rdma_active": len(up),
                "rdma_gbps": sum(dev["rate_gbps"] for dev in up),
                "free_tb": round(
                    sum(fs["free_gb"] for fs in detail["storage"]) / 1024, 2
                ),
            }
        )
        for gpu in host.gpus:
            state = "spare" if gpu.spare else "claimed" if gpu.claimed else "busy"
            data["gpus"].append(
                {
                    "gpu": f"{host.hostname}:{gpu.index}",
                    "host": host.hostname,
                    "index": gpu.index,
                    "model": gpu.name.removeprefix("NVIDIA "),
                    "used_gb": round(gpu.used_mib / 1024, 1),
                    "total_gb": round(gpu.total_mib / 1024, 1),
                    "util_pct": gpu.util_pct,
                    "state": state,
                }
            )
        for fs in detail["storage"]:
            data["storage"].append(
                {"fs": f"{host.hostname}:{fs['mount']}", "host": host.hostname, **fs}
            )
        for dev in detail["rdma"]:
            data["rdma_devices"].append(
                {
                    "dev": f"{host.hostname}:{dev['device']}",
                    "host": host.hostname,
                    **dev,
                }
            )
        details.append(
            {
                "alias": host.alias,
                "host": host.hostname,
                "ip": detail["ip"],
                "rdma": {
                    dev["netdev"]: (dev["device"], dev["gid_index"])
                    for dev in up
                    if dev["gid_index"] >= 0
                },
            }
        )
    if not args.skip_links:
        data["rdma_links"] = link_rows(details, args.timeout, args.ssh_config)
    return data


def main() -> int:
    """Parse arguments, snapshot the pool and write the row files.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="directory for the JSON files")
    parser.add_argument("--skip-links", action="store_true", help="no RDMA pingpong")
    parser.add_argument("--max-used-mib", type=int, default=1024)
    parser.add_argument("--max-util", type=int, default=10)
    parser.add_argument("--pattern", default=DEFAULT_PATTERNS)
    parser.add_argument("--exclude", default=DEFAULT_EXCLUDE)
    parser.add_argument("--ssh-config", default=DEFAULT_SSH_CONFIG)
    parser.add_argument("--timeout", type=int, default=8)
    args = parser.parse_args()

    data = collect(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in data.items():
        (out / f"{name}.json").write_text(json.dumps(rows, indent=1) + "\n")
        print(f"{name}.json: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
