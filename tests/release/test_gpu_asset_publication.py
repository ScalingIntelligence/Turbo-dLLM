from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _publish(tmp_path: Path, *, mismatch: bool = False):
    assets = tmp_path / "assets"
    assets.mkdir()
    wheel = assets / "turbo_dllm-native.whl"
    wheel.write_bytes(b"qualified wheel")
    (assets / "gpu-sm90.json").write_bytes(b"manifest")
    (assets / "gpu-bundles.json").write_bytes(b"catalog")
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "name": wheel.name,
                        "digest": "sha256:"
                        + (
                            "0" * 64
                            if mismatch
                            else hashlib.sha256(wheel.read_bytes()).hexdigest()
                        ),
                    }
                ],
                "uploads": [],
            }
        )
    )
    executable = tmp_path / "gh"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import hashlib, json, os, pathlib, sys\n"
        "state = pathlib.Path(os.environ['FAKE_GH_STATE'])\n"
        "data = json.loads(state.read_text())\n"
        "if sys.argv[1:3] == ['release', 'view']:\n"
        "    print('123')\n"
        "elif sys.argv[1] == 'api':\n"
        "    if sys.argv[2] != 'repos/ScalingIntelligence/Turbo-dLLM/releases/123':\n"
        "        sys.exit(1)\n"
        "    print(json.dumps(data))\n"
        "elif sys.argv[1:3] == ['release', 'upload']:\n"
        "    path = pathlib.Path(sys.argv[-1])\n"
        "    if any(a['name'] == path.name for a in data['assets']):\n"
        "        sys.exit(1)\n"
        "    data['uploads'].append(path.name)\n"
        "    data['assets'].append({'name': path.name, 'digest': 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()})\n"
        "    state.write_text(json.dumps(data))\n"
        "else:\n"
        "    sys.exit(2)\n"
    )
    executable.chmod(0o755)
    command = [
        sys.executable,
        str(ROOT / ".github/scripts/publish_gpu_assets.py"),
        "--directory",
        str(assets),
        "--release",
        "v0.1.1",
        "--repository",
        "ScalingIntelligence/Turbo-dLLM",
    ]
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GH_STATE": str(state),
    }
    completed = subprocess.run(command, capture_output=True, text=True, env=env)
    return completed, json.loads(state.read_text()), command, env


def test_partial_upload_resumes_with_catalog_last_and_can_be_repeated(tmp_path: Path):
    completed, state, command, env = _publish(tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert state["uploads"] == ["gpu-sm90.json", "gpu-bundles.json"]
    repeated = subprocess.run(command, capture_output=True, text=True, env=env)
    assert repeated.returncode == 0, repeated.stderr
    assert (
        json.loads(Path(env["FAKE_GH_STATE"]).read_text())["uploads"]
        == state["uploads"]
    )


def test_mismatched_existing_asset_blocks_all_uploads(tmp_path: Path):
    completed, state, _, _ = _publish(tmp_path, mismatch=True)
    assert completed.returncode != 0
    assert "differs" in completed.stderr
    assert state["uploads"] == []
