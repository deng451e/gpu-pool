---
name: gpu-pooling
description: Find spare GPUs of a required model (H200, H100, RTX PRO 6000, ...) across the GPU servers listed in ~/.ssh/config, claim them, and dispatch a test or benchmark there. Use before any GPU test, benchmark or inference-server launch when the task needs a specific GPU type or count, when the local GPUs are busy, or when asked which GPUs are free.
allowed-tools: Bash, Read, Grep, Glob
argument-hint: "[--gpu h200|pro6000] [--count N]"
---

# GPU pooling: pick a spare GPU on a shared server, then run there

Shared GPU servers often have no scheduler, so "which GPU do I use" is answered by looking,
every time. `scripts/gpu_pool.py` looks at this machine plus every pool host in
`~/.ssh/config` in parallel and reports the spare GPUs. Paths below are relative to the
folder that holds this file (for example `~/.claude/skills/gpu-pooling`).

## Pool membership lives in `~/.ssh/config`

Every `Host` alias matching `gpu-*` is a pool member, except CI servers: an alias with `ci`
as its own name component (`gpu-ci-01`, `lab-ci2`) is skipped, because CI runners are not
test capacity. Override with `--pattern "lab-*,gpu-*"` / `$GPU_POOL_PATTERN`
(comma-separated globs) and `--exclude <regex>` / `$GPU_POOL_EXCLUDE`. The config holds
connection details only; the GPU model is read live from `nvidia-smi`, so nothing about
the hardware has to be encoded in the alias.

```sshconfig
Host gpu-node-01
    HostName <address>
Host gpu-node-02
    HostName <address>
    Port <port>
Host gpu-*
    User <user>
    IdentityFile ~/.ssh/<key>
    IdentitiesOnly yes
```

Hosts must accept key auth without a prompt (`BatchMode=yes`) and already be in
`known_hosts`. To add a server: append a `Host` block and `ssh` to it once by hand. This
machine is always included as `local` when it has `nvidia-smi`, and an alias that points
back at it (or two aliases for one machine) is listed once. Do not edit `~/.ssh/config`
yourself; give the user the block to add. Never run pool work on a CI server, even if asked
for "any free GPU".

## Steps: dispatch a task to spare GPUs

1. **Derive the requirement from the task**: GPU model (`h200`, `pro6000`, or none if any
   will do) and count. Model matters when the task names hardware, compares against earlier
   numbers taken on a model, or needs the memory (H200: 141 GB; RTX PRO 6000: 96 GB).
   The filter is a case- and space-insensitive substring of the `nvidia-smi` name.
2. **Look**: `python3 scripts/gpu_pool.py` prints every GPU with `SPARE` / `busy` /
   `claimed`; unreachable hosts are listed, not fatal. `--json` for the same data.
3. **Pick and claim**:
   ```bash
   python3 scripts/gpu_pool.py --gpu h200 --count 2 --claim
   ```
   ```text
   HOST=gpu-node-02
   HOSTNAME=gpu-node-02
   GPU_MODEL=NVIDIA H200
   CUDA_VISIBLE_DEVICES=3,4
   ```
   `local` wins when it qualifies (no SSH hop); otherwise the host with the most spare
   matching GPUs. Exit code 1 means nothing qualifies: report that with the table from
   step 2 and stop. Never fall back to a busy GPU or a different model on your own.
4. **Run** with exactly those indices:
   - `HOST=local`: `CUDA_VISIBLE_DEVICES=3,4 <command>`.
   - remote: ship the code first (below), then start the job detached so it survives the
     SSH session, and poll it:
     ```bash
     ssh gpu-node-02 'tmux new -d -s <name> "cd <dir> && CUDA_VISIBLE_DEVICES=3,4 <command> 2>&1 | tee <name>.log"'
     ssh gpu-node-02 'tail -n 20 <dir>/<name>.log'
     ```
5. **Verify, then clean up**: once the job has loaded, `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv`
   on the host must show only your PIDs on those GPUs. If someone else landed there too,
   stop your job and pick again. When done, kill your processes and the tmux session, and
   confirm the GPUs read as spare again.

## What "spare" means

At most 1024 MiB in use **and** at most 10% utilization (`--max-used-mib`, `--max-util`),
and not claimed. Both conditions matter: idle-but-loaded GPUs belong to someone's server,
and a GPU can show 100% utilization with almost no memory in use.

`--claim` touches `/tmp/gpu-pool-claims/<index>` on the chosen host; a claim hides the GPU
from other `gpu_pool.py` runs for 10 minutes, which covers the gap until your job has
allocated memory. It only coordinates agents and people that use this script. Others do
not see claims, which is why step 5 exists.

## Running on a remote host

Home directories are usually per-host local disks, not shared: a checkout of your project
on the remote host is a separate copy, often on an older commit, with its own virtualenvs
(or none). Check before assuming: `ssh <host> 'git -C <repo> log --oneline -1; ls <venvs>'`.

- **Code**: do not touch a working tree other people or agents use on the host. Copy the
  tree under test to a scratch directory. `rsync` is not installed everywhere, so use:
  ```bash
  git archive --format=tar HEAD | ssh <host> 'mkdir -p ~/scratch/<name> && tar -x -C ~/scratch/<name>'
  ```
  `git archive` ships committed files only; commit (or `git stash create` and archive that
  hash) to include local edits.
- **Environment**: reuse an existing remote virtualenv with `PYTHONPATH=~/scratch/<name>`
  for pure-Python changes. Native extensions need a build on that host; start long
  installs in the background.
- **Models**: `HF_HOME` caches are per host too. Check the model is present before
  launching; copy with `tar | ssh tar` rather than re-downloading.
- **Disk**: check `df -h ~` before copying models or building environments, and remove
  your scratch directory afterwards.
- **Ports**: the remote job listens on the remote host. Reach it with
  `ssh -L <port>:localhost:<port> <host>` or run the client on the same host.

## Pool snapshot: GPUs, RDMA interconnect, storage

`scripts/pool_report.py --out <dir>` probes every pool host once and writes five JSON row
files:

| File | One row per | Source |
| --- | --- | --- |
| `hosts.json` | host | reachability, GPU model, spare count, CPU and memory usage, RDMA and disk totals |
| `gpus.json` | GPU | `nvidia-smi`, with the same spare/claimed/busy verdict as `gpu_pool.py` |
| `storage.json` | local filesystem | `df` (tmpfs, overlay and EFI partitions left out) |
| `rdma_devices.json` | RDMA device | sysfs: rate, port state, link layer, NUMA node |
| `rdma_links.json` | host pair and interface | `ibv_rc_pingpong` result and round-trip time |

The link check runs a 1000-message `ibv_rc_pingpong` between the same-named network
interface (for example `bond0`, `rail1`; RDMA device names can differ between hosts) on
every pair of hosts, over the first routable RoCE v2 GID, on TCP ports from 18600. Hosts at
different sites share no interface names, so only same-site pairs are checked. It proves
the RDMA path works; it is not a throughput number (use `ib_write_bw` for that, and only on
GPUs and NICs nobody else is using). `--skip-links` leaves it out.

Use the snapshot to choose where to run: a multi-host test needs two hosts whose link rows
pass, and a test that downloads models or builds an environment needs a filesystem with room.

## Live views

**Terminal.** `scripts/pool_watch.py` redraws GPU, CPU, memory and disk usage for every
pool host once a second (`--interval` to change it, Ctrl-C to stop). It holds one SSH
connection per host and streams from it; an unreachable host is shown with the SSH error
and retried every 5 s. `--once` prints a single frame and exits.

**Web dashboard.** `scripts/pool_serve.py` serves the same per-second data as a web page on
`127.0.0.1:8765` (`--port`, `--host`) and opens it in a browser (`--no-browser` skips
that). It needs only Python 3.9+ and key-based `ssh`, so it also runs on a laptop. It has
no authentication: keep it on localhost and reach it from another machine through
`ssh -L 8765:localhost:8765 <host>`.

- **Choosing servers**: the page's **Servers** button lists every host of an SSH config
  (`~/.ssh/config` or any path typed there) and changes the tracked set live; the choice is
  saved in `~/.gpu-pool/selection.json` and reused on the next start. `--hosts 'gpu-*'`
  overrides it for one run and `--ssh-config` its config path. Before anything is saved,
  the default set is the pool hosts minus the aliases or globs in `~/.gpu-pool/hide`. The
  selection API only answers requests addressed to 127.0.0.1/localhost from the page
  itself (Host/Origin checks, JSON-only POST).
- **What it shows**: per GPU, memory in use; Busy (`nvidia-smi` utilization, the share of
  time any kernel ran, so one tiny kernel reads 100%); SM and TC, the share of time
  streaming multiprocessors and tensor cores were actually active (DCGM profiling fields
  1002/1004 from one background `dcgmi dmon` per connection, which exits with it; needs a
  running DCGM host engine, otherwise power draw is the best load signal); and power
  against the board limit. Per host it also shows CPU per NUMA node with one square per
  CPU, host memory, DRAM traffic per socket, and storage. DRAM traffic comes from resctrl
  (`mbm_total_bytes`), which exists only after root runs
  `mount -t resctrl resctrl /sys/fs/resctrl`; without it the page says so.
- **Running it detached**: `setsid nohup python3 scripts/pool_serve.py --no-browser > pool_serve.log 2>&1 &`;
  stop it via the PID listening on its port (`ss -ltnp 'sport = :8765'`), because
  `pkill -f` can match the shell that runs it.
- **macOS app**: `python3 mac/build_installer.py --out install.sh && bash install.sh`
  installs the `dashboard` command (`dashboard stop` ends a background server) and a
  **GPU Dashboard** app on the Desktop. The installer is self-contained, so it can be copied
  to other Macs. Every alias must work as plain `ssh <alias>` from the Mac (VPN or
  `ProxyJump` for private addresses).

## Report

State which host, GPU model and indices the run used, so the numbers can be reproduced on
the same hardware.
