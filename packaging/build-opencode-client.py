#!/usr/bin/env python3
"""Build the Recursive-branded OpenCode client that rec launch opencode downloads."""

import argparse
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "cli/clients/opencode"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bun", default="bun", help="Bun 1.4.2 executable (default: %(default)s)")
    parser.add_argument(
        "--source", type=Path, help="existing clean OpenCode checkout at the pinned commit"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist/opencode-darwin-arm64.gz",
        help="gzipped client to publish (default: dist/opencode-darwin-arm64.gz)",
    )
    args = parser.parse_args()
    pins = json.loads((ASSETS / "pins.json").read_text())
    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        parser.error("this preview builds on Apple Silicon Macs only")
    bun = shutil.which(args.bun)
    if bun is None:
        parser.error("install Bun 1.4.2 or provide --bun PATH")
    bun = str(Path(bun).resolve())
    if subprocess.check_output([bun, "--version"], text=True).strip() != pins["bun"]:
        parser.error(f"Bun {pins['bun']} is required")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}; choose a new --output")
    source = args.source.resolve() if args.source else ROOT / ".build/opencode-client"
    if args.source is None:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                f"v{pins['version']}",
                pins["repository"],
                str(source),
            ],
            check=True,
        )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if head != pins["commit"] or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=source
    ):
        parser.error("source must be clean and match pins.json exactly")
    for patch in ("logo-slots.patch", "rec-home.patch"):
        subprocess.run(["git", "apply", str(ASSETS / patch)], cwd=source, check=True)
    shutil.copytree(
        ASSETS / "recursive-logo", source / "packages/tui/test/fixture/plugin/recursive-logo"
    )
    shutil.copy2(
        ASSETS / "recursive-logo.test.tsx", source / "packages/tui/test/recursive-logo.test.tsx"
    )
    env = dict(
        os.environ,
        PATH=str(Path(bun).parent) + os.pathsep + os.environ["PATH"],
        OPENCODE_VERSION=pins["version"],
        OPENCODE_CHANNEL="latest",
    )
    subprocess.run([bun, "install", "--frozen-lockfile"], cwd=source, env=env, check=True)
    subprocess.run([bun, "run", "check"], cwd=source, env=env, check=True)
    subprocess.run(
        [bun, "test", "test/recursive-logo.test.tsx"],
        cwd=source / "packages/tui",
        env=env,
        check=True,
    )
    subprocess.run(
        [bun, "run", "script/build.ts", "--single"],
        cwd=source / "packages/cli",
        env=env,
        check=True,
    )
    # rec embeds the logo plugin; the release asset is only the patched client.
    output.parent.mkdir(parents=True, exist_ok=True)
    binary = (source / "packages/cli/dist/cli-darwin-arm64/bin/opencode").read_bytes()
    output.write_bytes(gzip.compress(binary, mtime=0))
    data = output.read_bytes()
    print(
        json.dumps(
            {"asset": output.name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    )


if __name__ == "__main__":
    main()
