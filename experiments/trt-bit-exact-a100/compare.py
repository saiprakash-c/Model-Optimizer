# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare TensorRT outputs between GPUs, bit for bit (run on Thor).

Usage: python compare.py <results_dir>
<results_dir>/<tag>/ holds meta.json from run_trt.py, one folder per GPU. Outputs are compared by
the sha256 of their bytes; <engine>.npy files, when present, give the size of any difference.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


def main(results):
    """Print a table of run-to-run and cross-GPU equality per engine."""
    tags = sorted(p.name for p in results.iterdir() if (p / "meta.json").exists())
    metas = {t: json.loads((results / t / "meta.json").read_text()) for t in tags}
    for t in tags:
        print(f"{t}: {metas[t]['gpu']} | TensorRT {metas[t]['tensorrt']}")
    engines = sorted(set.intersection(*(set(m["engines"]) for m in metas.values())))

    print(
        "\n| engine | "
        + " | ".join(f"runs equal on {t}" for t in tags)
        + " | "
        + " | ".join(f"{a} vs {b}" for a, b in combinations(tags, 2))
        + " |"
    )
    print("|---" * (1 + len(tags) + len(list(combinations(tags, 2)))) + "|")
    all_exact = True
    for e in engines:
        row = [e]
        for t in tags:
            runs = metas[t]["engines"][e]["output_sha256_per_run"]
            ok = len(set(runs)) == 1
            all_exact &= ok
            row.append(f"{'yes' if ok else 'NO'} ({len(runs)} runs)")
        for a, b in combinations(tags, 2):
            assert (
                metas[a]["engines"][e]["engine_sha256"] == metas[b]["engines"][e]["engine_sha256"]
            ), f"{e}: different engine files on {a} and {b}"
            if (
                metas[a]["engines"][e]["output_sha256_per_run"][0]
                == metas[b]["engines"][e]["output_sha256_per_run"][0]
            ):
                row.append("bit-exact")
                continue
            all_exact = False
            fa, fb = results / a / f"{e}.npy", results / b / f"{e}.npy"
            if not (fa.exists() and fb.exists()):
                row.append("DIFFERENT")
                continue
            diff = np.abs(np.load(fa).astype(np.float64) - np.load(fb).astype(np.float64))
            row.append(f"DIFFERENT ({(diff > 0).mean():.2%} elems, max abs {diff.max():.3g})")
        print("| " + " | ".join(row) + " |")
    print(f"\nengines compared: {len(engines)}")
    print("RESULT: BIT-EXACT" if all_exact else "RESULT: NOT BIT-EXACT")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
