#!/usr/bin/env python3
"""Validate a d4r preset-M layer capture against the pinned d4r model.

A Windows capture contains one directory per whitelisted launch
(`rrlite_enc1_4x4`, `rrlite_enc2_4x4`, six `rrlite_enc3_tube_4x4`, `rrlite_dec2_4x4`,
`rrlite_dec1_4x4`). This script parses those launches with d4r's own
`model_enc3.Params`, extracts the plain tensors with `m_model.plain_layer`, and
reports their shapes, finiteness, byte-identity hashes, and (with `--compare`)
whether two captures carry identical weights.

usage: validate_m_capture.py CAPTURE [--compare OTHER_CAPTURE]
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "references" / "sources" / "d4r" / "engine"))

import m_model as mm  # noqa: E402
import model_enc3 as me  # noqa: E402

# Network order, including the six tube launches.
ORDER = ["enc1", "enc2"] + ["enc3_tube"] * 6 + ["dec2", "dec1"]


def captures(capture):
    """Map (layer, instance) -> replay directory, in network order."""
    found = {}
    for directory in Path(capture).iterdir():
        if not directory.is_dir():
            continue
        name = directory.name
        if "-rrlite_" not in name:
            continue
        kernel = name.split("-rrlite_", 1)[1]
        if not kernel.endswith("_4x4"):
            continue
        layer = kernel[:-4]
        if layer not in mm.LAYERS:
            continue
        instance = int(name.split("-")[1])
        found[(layer, instance)] = directory
    out = []
    queues = {}
    for (layer, _), directory in found.items():
        queues.setdefault(layer, []).append(directory)
    for layer in queues:
        queues[layer].sort(key=lambda d: int(d.name.split("-")[1]))
    for layer in ORDER:
        if not queues.get(layer):
            raise SystemExit(f"missing capture for {layer}")
        out.append((layer, queues[layer].pop(0)))
    leftovers = sum(len(queue) for queue in queues.values())
    if leftovers > 0:
        raise SystemExit(f"{leftovers} extra M layer capture(s) not consumed by the network order")
    return out


def digest(tensors):
    h = hashlib.sha256()
    for key in sorted(tensors):
        value = tensors[key]
        values = value if isinstance(value, list) else [value]
        for item in values:
            array = np.ascontiguousarray(item)
            h.update(key.encode())
            h.update(str(array.shape).encode())
            h.update(array.astype(np.float16).tobytes())
    return h.hexdigest()


def describe(capture):
    result = []
    for layer, directory in captures(capture):
        P = me.Params(str(directory))
        tensors = mm.plain_layer(P, layer)
        arrays = []
        for key, value in tensors.items():
            values = value if isinstance(value, list) else [value]
            arrays.extend(values)
        finite = all(np.isfinite(a).all() for a in arrays)
        nonzero = sum(int(np.count_nonzero(a)) for a in arrays)
        elements = sum(int(a.size) for a in arrays)
        result.append({
            "layer": layer,
            "kernel": P.kernel,
            "grid": P.grid,
            "tokens": (P.W, P.H),
            "shift": (P.sx, P.sy),
            "C": mm.LAYERS[layer][0],
            "heads": mm.LAYERS[layer][1],
            "tensors": len(arrays),
            "elements": elements,
            "nonzero": nonzero,
            "finite": finite,
            "sha256": digest(tensors),
            "dir": directory,
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()

    rows = describe(args.capture)
    for row in rows:
        print(
            f"{row['layer']:10s} {row['kernel']:24s} grid={row['grid']} tokens={row['tokens']} "
            f"shift={row['shift']} C={row['C']} heads={row['heads']} "
            f"tensors={row['tensors']} elements={row['elements']} nonzero={row['nonzero']} "
            f"finite={row['finite']} sha256={row['sha256'][:16]}"
        )

    if not args.compare:
        return 0

    other = describe(args.compare)
    if len(other) != len(rows):
        raise SystemExit("capture layer counts differ")
    differences = 0
    for left, right in zip(rows, other):
        if left["layer"] != right["layer"]:
            raise SystemExit(f"layer order differs: {left['layer']} vs {right['layer']}")
        same = left["sha256"] == right["sha256"]
        differences += not same
        print(f"compare {left['layer']:10s} identical={same} {left['sha256'][:16]} {right['sha256'][:16]}")
    print(f"identical_layers={len(rows) - differences}/{len(rows)}")
    return 0 if differences == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
