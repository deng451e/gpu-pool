#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the one-file Mac installer for the live dashboard.

Embeds ``scripts/`` (as a gzipped tar) and ``dashboard_icon.png`` into
``install_template.sh``. Rerun after changing any dashboard script, then run
``bash <out>`` on the Mac (it is self-contained, so it can be copied to other Macs).
"""

# Standard
from pathlib import Path
import argparse
import base64
import io
import sys
import tarfile

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"


def scripts_archive() -> bytes:
    """Pack ``scripts/`` without bytecode caches.

    Returns:
        A gzipped tar whose single top-level entry is ``scripts/``.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(
            SCRIPTS,
            arcname="scripts",
            filter=lambda info: None if "__pycache__" in info.name else info,
        )
    return buffer.getvalue()


def main() -> int:
    """Write the installer.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="installer path to write")
    args = parser.parse_args()
    text = (HERE / "install_template.sh").read_text()
    text = text.replace(
        "@@SCRIPTS@@", base64.encodebytes(scripts_archive()).decode().rstrip()
    )
    icon = (HERE / "dashboard_icon.png").read_bytes()
    text = text.replace("@@ICON@@", base64.encodebytes(icon).decode().rstrip())
    out = Path(args.out)
    out.write_text(text)
    out.chmod(0o755)
    print(f"wrote {out} ({len(text) // 1024} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
