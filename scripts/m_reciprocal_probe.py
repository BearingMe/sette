#!/usr/bin/env python3
"""Measure D3D12 rsqrt/rcp on all widened f16 arguments (local diagnostic LUT)."""
import argparse
import subprocess
from pathlib import Path

import numpy as np

import m_layer_fixture as lf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dxc', default=r'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    root = Path(lf.REPO)
    subprocess.run([a.dxc, '-T', 'cs_6_2', '-E', 'main', '-O3',
                    str(root / 'src/native/reciprocal_probe_cs.hlsl'),
                    '-Fo', str(a.out / 'probe.cso')], check=True)
    (a.out / 'in.bin').write_bytes(np.arange(65536, dtype=np.uint32).tobytes())
    (a.out / 'params.bin').write_bytes(bytes(16))
    (a.out / 'graph.txt').write_text('''version 1
resource input 262144 in.bin
resource output 524288
resource scratch 16
stage reciprocal
shader probe.cso
threads 256 1 1
groups 256 1 1
params params.bin 16
srv 0 input
srv 1 input
srv 2 input
uav 0 output
uav 1 scratch
dump output 0 524288 reciprocal.bin
''')
    lf.run(str(a.out), str(root / 'build/native/Release/m_network_stage.exe'),
           str(a.out), 0, str(a.out / 'results'))
    values = np.fromfile(a.out / 'results/reciprocal.bin', np.float32).reshape(65536, 2)
    x = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float64)
    with np.errstate(all='ignore'):
        exact = np.stack([1 / np.sqrt(x), 1 / x], axis=1).astype(np.float32)
    # Positive finite nonzero arguments are the network's domain.
    sl = slice(1, 0x7c00)
    for i, name in enumerate(('rsqrt', 'rcp')):
        with np.errstate(over='ignore'):
            half_diff = int((values[sl, i].astype(np.float16) !=
                             exact[sl, i].astype(np.float16)).sum())
        print(name, 'f32 differences', int((values[sl, i] != exact[sl, i]).sum()),
              'f16 differences', half_diff)


if __name__ == '__main__':
    main()
