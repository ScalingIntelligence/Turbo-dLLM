"""Exercise the installed client with verified local, not-yet-published assets."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

from dllm_parallel.core.kernels.bundle_manifest import install_bundle


def main() -> None:
    manifest = Path(sys.argv[1])
    parsed = json.loads(manifest.read_text())
    assets = {
        item["url"]: manifest.parent / item["name"]
        for item in (*parsed["artifacts"], *parsed["runtime_artifacts"])
    }

    def open_asset(request, **kwargs):
        return io.BytesIO(assets[request.full_url].read_bytes())

    with patch("urllib.request.urlopen", side_effect=open_asset):
        install_bundle(manifest)


if __name__ == "__main__":
    main()
