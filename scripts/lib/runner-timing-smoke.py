"""Run the timing smoke preset against one visible GPU model in a private copy."""

import csv
import io
import json
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp


def visible_gpu_name(output: str) -> str:
    rows = list(csv.reader(io.StringIO(output)))
    if not rows or any(len(row) != 1 or not row[0].strip() for row in rows):
        raise ValueError("timing smoke could not identify visible GPUs")
    names = {row[0].strip() for row in rows}
    if len(names) != 1:
        raise ValueError(
            "timing smoke requires one GPU model; select devices with VIBESIM_RUNNER_GPUS"
        )
    return names.pop()


def run(root: Path) -> None:
    root = root.resolve()
    devices = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    gpu = visible_gpu_name(devices.stdout)
    presets = root / "presets"
    config = json.loads((presets / "predict_llama3_8b_iter.json").read_text())
    config["gpu"] = gpu
    (root / "logs").mkdir(exist_ok=True)
    logs = Path(mkdtemp(prefix="runner-timing-smoke-", dir=root / "logs"))
    config["log_dir"] = str(logs)
    # Keep the generated config beside the preset so cases_file stays relative.
    with NamedTemporaryFile(
        mode="w", suffix=".json", prefix=".runner-timing-", dir=presets
    ) as stream:
        json.dump(config, stream)
        stream.flush()
        print(f"timing smoke: gpu={gpu} config={stream.name} logs={logs}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "launcher", "timing-predict", stream.name],
            cwd=root,
            check=True,
        )
    report = logs / "reports/iter_breakdown.ans"
    if not report.is_file() or report.stat().st_size == 0:
        raise RuntimeError("timing-predict did not produce reports/iter_breakdown.ans")


if __name__ == "__main__":
    run(Path.cwd())
