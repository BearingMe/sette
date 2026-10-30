#!/usr/bin/env python3
"""B5.2: export one captured neural layer as a native D3D12 graph fixture.

The fixture is a graph manifest for `m_network_stage`, the init data for the
resources that come from the capture (the input field, the skip field, the
weights), the per-dispatch parameter blocks, and the captured output as the
reference.

Nothing here reinterprets a captured allocation: the input field is the captured
e4m3 bytes at the layer's own pointers, and the weights are the captured e4m3
matrices decoded losslessly to f32 after the natural-order permutation the block
runner expects (checked against the kernel-order oracle, EXP-019).

Subcommands:

  export CAPTURE --layer NAME --out DIR [--frame N]
      Write the graph, resources, parameters and references.
  run DIR [--probe EXE] [--shaders DIR] [--seed N] [--out RESULTS]
      Execute the graph.
  check DIR [--results DIR] [--json OUT]
      Compare every declared output, including the written-region test: two
      executions with different sentinel seeds must change *every* byte of a
      declared output and nothing outside it.

Fixture data is NVIDIA-derived; keep it in the temporary workspace.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import m_layer_oracle as lo      # noqa: E402
import m_model as mm             # noqa: E402
import model_enc3 as me          # noqa: E402

F16, F32, F64 = np.float16, np.float32, np.float64

PARAM_UINTS = 40
SHADERS = {
    'prepare': 'swin_prepare.cso', 'norm': 'swin_norm.cso',
    'matmul': 'swin_matmul.cso', 'attention': 'swin_attention.cso',
    'mlp': 'swin_mlp.cso', 'store': 'swin_store.cso', 'merge': 'swin_merge.cso',
}


def fail(message):
    raise SystemExit(f'error: {message}')


def layer_directory(capture, label, frame=1):
    dirs = lo.layers_of(capture)
    labels = ['enc1', 'enc2'] + [f'tube#{i}' for i in range(1, 7)] + ['dec2', 'dec1']
    if label not in labels:
        fail(f'unknown layer {label!r}; expected one of {labels}')
    index = labels.index(label)
    return dirs[index + 10 * (frame - 1)]


def padded_geometry(W, H, sx, sy):
    gw = ((W + sx + 7) // 8) * 8
    gh = ((H + sy + 7) // 8) * 8
    return gw, gh


class WeightPack:
    """The block's matrices and vectors in natural order, one f32 buffer."""

    def __init__(self, P, name):
        self.P, self.name = P, name
        C, heads, NPM, CIN = mm.LAYERS[name]
        m = lo.natural_matrices(P, name)
        t = m['t']
        self.parts = {}
        self.offset = {}
        cursor = 0
        def add(key, array, dtype=F32):
            nonlocal cursor
            a = np.ascontiguousarray(np.asarray(array, dtype))
            self.parts[key] = a
            self.offset[key] = cursor
            cursor += a.size
        add('g1', t['g1'])
        add('bo', t['bo'])
        add('g2', t['g2'])
        add('b2', t['b2'])
        add('q', m['q'])
        add('v', m['v'])
        add('wo', m['wo'])
        add('w1', np.concatenate(m['w1'], axis=0) if m['w1'] else np.zeros(0))
        add('b1', np.concatenate(m['b1']) if m['b1'] else np.zeros(0))
        add('w2', np.concatenate(m['w2'], axis=0) if m['w2'] else np.zeros(0))
        if NPM:
            add('mergeW', t['merge_w'])
            add('mergeB', t['merge_b'])
        if CIN:
            add('expandW', t['expand_w'])
            add('expandB', t['expand_b'])
        add('bias', np.concatenate(t['bias']))
        self.total = cursor
        self.data = np.concatenate([self.parts[k].ravel() for k in self.parts])


class GraphWriter:
    def __init__(self, out, name, P, W, H, C, heads, NPM, CIN, sx, sy):
        self.out = out
        self.name = name
        self.P = P
        self.W, self.H = W, H
        self.C, self.heads, self.NPM, self.CIN = C, heads, NPM, CIN
        self.sx, self.sy = sx, sy
        self.GW, self.GH = padded_geometry(W, H, sx, sy)
        self.tokens = self.GW * self.GH
        self.lowW, self.lowH = W // 2, H // 2
        self.lines = ['version 1']
        self.resources = []
        self.stages = []
        self.dumps = []
        self.record = {'layer': name, 'geometry': {
            'W': W, 'H': H, 'C': C, 'heads': heads, 'NPM': NPM, 'CIN': CIN,
            'sx': sx, 'sy': sy, 'GW': self.GW, 'GH': self.GH,
            'lowW': self.lowW, 'lowH': self.lowH, 'tokens': self.tokens,
        }, 'outputs': {}, 'stages': []}

    def resource(self, name, bytes_, init=None):
        if bytes_ % 4:
            bytes_ += 4 - bytes_ % 4
        if init is not None:
            if isinstance(init, (bytes, bytearray)):
                init = np.frombuffer(init, np.uint8)
            else:
                init = np.ascontiguousarray(init, np.uint8)
            if init.size != bytes_:
                raise SystemExit(f'{name}: init {init.size} B, resource {bytes_} B')
            path = f'{name}.bin'
            open(os.path.join(self.out, path), 'wb').write(init.tobytes())
            self.lines.append(f'resource {name} {bytes_} {path}')
        else:
            self.lines.append(f'resource {name} {bytes_}')
        self.resources.append((name, bytes_))
        return name

    def f16_resource(self, name, values):
        return self.resource(name, int(np.asarray(values, F16).size * 2),
                             np.asarray(values, F16).tobytes())

    def stage(self, name, kind, groups, threads=(64, 1, 1), params=None):
        self.stages.append({'name': name, 'kind': kind, 'groups': list(groups),
                            'threads': list(threads)})
        self.lines.append(f'stage {name}')
        self.lines.append(f'shader {SHADERS[kind]}')
        self.lines.append('threads {} {} {}'.format(*threads))
        self.lines.append('groups {} {} {}'.format(*groups))
        self.lines.append(f'params params-{name}.bin {PARAM_UINTS * 4}')
        return name

    def bind(self, kind, reg, resource):
        if kind == 'srv':
            self.lines.append(f'srv {reg} {resource}')
        else:
            self.lines.append(f'uav {reg} {resource}')

    def bind_stage(self, srvs, uavs, pad_srv, pad_uav):
        """Every dispatch declares t0..t2 and u0..u1, padded with scratch.

        The root signature covers the shader's full register set, so a stage
        that reads fewer resources binds a harmless unused descriptor instead of
        producing a signature that no pipeline can use.
        """
        for i in range(3):
            self.bind('srv', i, srvs[i] if i < len(srvs) else pad_srv)
        for i in range(2):
            self.bind('uav', i, uavs[i] if i < len(uavs) else pad_uav)

    def dump(self, resource, offset, bytes_, file):
        self.lines.append(f'dump {resource} {offset} {bytes_} {file}')
        self.dumps.append({'resource': resource, 'offset': offset, 'bytes': bytes_,
                           'file': file})

    def finish(self):
        open(os.path.join(self.out, 'graph.txt'), 'w').write('\n'.join(self.lines) + '\n')


def params_block(g, offsets, **kwargs):
    """The shared cbuffer, with the per-dispatch values filled in."""
    C = g.C
    block = [0] * PARAM_UINTS
    block[0] = C
    block[1] = g.heads
    block[2] = g.GW
    block[3] = g.GH
    block[4] = g.W
    block[5] = g.H
    block[6] = g.sx
    block[7] = g.sy
    block[8] = g.tokens
    block[9] = g.lowW
    block[10] = g.lowH
    block[11] = C // 8
    block[19] = offsets['g1']
    block[20] = offsets['bo']
    block[21] = offsets['g2']
    block[22] = offsets['b2']
    block[23] = offsets['q']
    block[24] = offsets['v']
    block[25] = offsets['wo']
    block[26] = offsets['w1']
    block[27] = offsets['b1']
    block[28] = offsets['w2']
    block[29] = offsets['bias']
    block[30] = offsets.get('mergeW', 0)
    block[31] = offsets.get('mergeB', 0)
    block[32] = offsets.get('expandW', 0)
    block[33] = offsets.get('expandB', 0)
    block[34] = g.NPM
    block[35] = C // 8
    for key, value in kwargs.items():
        if key == 'matrix':
            block[36] = offsets[value]
            continue
        index = {'mode': 12, 'srcFmt': 13, 'srcW': 14, 'srcH': 15, 'K': 16,
                 'N': 17, 'c0mode': 18}[key]
        block[index] = value
    return np.array(block, np.uint32)


def export(capture, label, out, frame):
    # A one-kernel capture (translated PTX) is already a replay directory.
    if os.path.basename(os.path.normpath(capture)).startswith('replay-'):
        directory = capture
    else:
        directory = layer_directory(capture, label, frame)
    name = 'enc3_tube' if label.startswith('tube') else label
    C, heads, NPM, CIN = mm.LAYERS[name]
    P, x, skip, ref, mref = lo.read_layer(directory, name)
    Praw, raw = lo.raw_layer(directory, name)
    W, H = P.W, P.H
    sx, sy = P.sx, P.sy
    os.makedirs(out, exist_ok=True)
    g = GraphWriter(out, label, P, W, H, C, heads, NPM, CIN, sx, sy)
    pack = WeightPack(P, name)
    offsets = pack.offset

    weights = g.resource('weights', pack.total * 4, pack.data.astype(F32).tobytes())
    x0 = g.resource('x0', g.tokens * C * 2)
    h1 = g.resource('h1', g.tokens * C * 2)
    res = g.resource('res', g.tokens * C * 2)
    q = g.resource('q', g.tokens * 32 * heads * 2)
    v = g.resource('v', g.tokens * 32 * heads * 2)
    o = g.resource('o', g.tokens * 32 * heads * 2)
    x1 = g.resource('x1', g.tokens * C * 2)
    h2 = g.resource('h2', g.tokens * C * 2)
    x2 = g.resource('x2', g.tokens * C * 2)
    x2b = g.resource('x2b', g.tokens * C * 2)
    outfield = g.resource('out', W * H * C)
    scratch = g.resource('scratch', 16)
    mergefield = None
    if NPM:
        mergefield = g.resource('merge', (W // 2) * (H // 2) * NPM)
    low = expand = skipr = None
    if CIN:
        low = g.resource('low', raw['low'].size, raw['low'])
        skipr = g.resource('skip', raw['skip'].size, raw['skip'])
        expand = g.resource('expand', g.lowW * g.lowH * 4 * C * 2)
    else:
        g.resource('in', raw['input'].size, raw['input'])

    def write_params(stage, bias=None, **kwargs):
        block = params_block(g, offsets, **kwargs)
        if bias is not None:
            block[29] = offsets[bias]
        open(os.path.join(out, f'params-{stage}.bin'), 'wb').write(block.tobytes())

    if CIN:
        stage = g.stage('expand', 'matmul',
                        [int(np.ceil(g.lowW * g.lowH * (4 * C // 16) / 64)), 1, 1])
        write_params(stage, bias='expandB', mode=1, srcFmt=1, srcW=g.lowW, srcH=g.lowH,
                     K=CIN, N=4 * C, c0mode=1, matrix='expandW')
        g.bind_stage([low, weights, weights], [expand], weights, scratch)

    stage = g.stage('prepare', 'prepare', [int(np.ceil(g.tokens / 64)), 1, 1])
    write_params(stage, mode=1 if CIN else 0)
    g.bind_stage([expand if CIN else 'in', skipr if CIN else weights, weights],
                 [x0], weights, scratch)

    stage = g.stage('norm1', 'norm', [int(np.ceil(g.tokens / 64)), 1, 1])
    write_params(stage, mode=0)
    g.bind_stage([x0, weights, weights], [h1, res], weights, scratch)

    for role, target in (('q', q), ('v', v)):
        stage = g.stage(role, 'matmul',
                        [int(np.ceil(g.tokens * (32 * heads // 16) / 64)), 1, 1])
        write_params(stage, K=C, N=32 * heads, c0mode=0, srcFmt=0, matrix=role)
        g.bind_stage([h1, weights, weights], [target], weights, scratch)

    stage = g.stage('attention', 'attention', [int(np.ceil(g.tokens / 64)), 1, 1])
    write_params(stage)
    g.bind_stage([q, v, weights], [o], weights, scratch)

    stage = g.stage('wo', 'matmul', [int(np.ceil(g.tokens * (C // 16) / 64)), 1, 1])
    write_params(stage, K=32 * heads, N=C, c0mode=2, srcFmt=0, matrix='wo')
    g.bind_stage([o, weights, res], [x1], weights, scratch)

    stage = g.stage('norm2', 'norm', [int(np.ceil(g.tokens / 64)), 1, 1])
    write_params(stage, mode=1)
    g.bind_stage([x1, weights, weights], [h2, x2], weights, scratch)

    stage = g.stage('mlp', 'mlp', [int(np.ceil(g.tokens / 64)), 1, 1])
    write_params(stage)
    g.bind_stage([h2, weights, x2], [x2b], weights, scratch)

    stage = g.stage('store', 'store', [int(np.ceil(g.tokens / 64)), 1, 1])
    write_params(stage)
    g.bind_stage([x2b], [outfield], weights, scratch)

    if NPM:
        merged_tokens = (g.GW // 2) * (g.GH // 2)
        stage = g.stage('merge', 'merge',
                        [int(np.ceil(merged_tokens * (NPM // 16) / 64)), 1, 1])
        write_params(stage)
        g.bind_stage([x2b, weights], [mergefield], weights, scratch)
        g.dump(mergefield, 0, (W // 2) * (H // 2) * NPM, 'out-merge.bin')
        g.record['outputs']['merge'] = {
            'bytes': (W // 2) * (H // 2) * NPM,
            'geometry': [W // 2, H // 2, NPM], 'kind': 'e4m3'}

    g.dump(outfield, 0, W * H * C, 'out-out.bin')
    g.record['outputs']['out'] = {'bytes': W * H * C, 'geometry': [W, H, C],
                                  'kind': 'e4m3'}
    g.finish()

    open(os.path.join(out, 'ref-out.bin'), 'wb').write(raw['output'].tobytes())
    if NPM:
        open(os.path.join(out, 'ref-merge.bin'), 'wb').write(raw['merge'].tobytes())

    g.record['kernel'] = os.path.basename(directory)
    g.record['weights'] = {'elements': int(pack.total),
                           'offsets': {k: int(v) for k, v in offsets.items()}}
    g.record['graph'] = 'graph.txt'
    g.record['stages'] = g.stages
    g.record['resources'] = g.resources
    with open(os.path.join(out, 'fixture.json'), 'w') as handle:
        json.dump(g.record, handle, indent=2)
    print(f'{label}: {W}x{H}x{C} shift {sx},{sy}; padded {g.GW}x{g.GH}, '
          f'{len(g.stages)} dispatches, {pack.total} f32 weight elements')
    for key, entry in g.record['outputs'].items():
        print(f'  {key}: {entry["bytes"]} B  {entry["geometry"]}')
    print(f'  -> {out}')
    return 0


SENTINEL = lambda i, seed: (i * 37 + 11 + seed * 97) & 0xFF


def run(fixture, probe, shaders, seed, results):
    command = [probe, '--graph', os.path.join(fixture, 'graph.txt'), '--out', results,
               '--shaders', shaders, '--seed', str(seed)]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f'error: {probe} failed on {fixture}:\n'
                         f'{done.stdout[-800:]}{done.stderr[-800:]}')
    return results


def compare_outputs(fixture, results):
    record = json.load(open(os.path.join(fixture, 'fixture.json')))
    rows, problems = [], []
    for role, entry in record['outputs'].items():
        path = os.path.join(results, f'out-{role}.bin')
        if not os.path.exists(path):
            problems.append(f'{role}: no out-{role}.bin')
            continue
        native = np.fromfile(path, np.uint8)
        if native.size != entry['bytes']:
            problems.append(f'{role}: {native.size} B, declared {entry["bytes"]} B')
            continue
        ref = np.fromfile(os.path.join(fixture, f'ref-{role}.bin'), np.uint8)
        if ref.size != entry['bytes']:
            problems.append(f'{role}: reference {ref.size} B, declared {entry["bytes"]} B')
            continue
        identical = int((native == ref).sum())
        rows.append({'role': role, 'bytes': native.size, 'identical': identical,
                     'fraction': identical / native.size})
        print(f'{role:6s} {native.size:>9d} B  identical {100 * identical / native.size:8.4f}%  '
              f'{native.size - identical} differ')
    return rows, problems


def written_region(fixture, results0, results1, role, entry):
    """Bytes an output actually wrote: not the sentinel under either seed."""
    n = entry['bytes']
    a = np.fromfile(os.path.join(results0, f'out-{role}.bin'), np.uint8)
    b = np.fromfile(os.path.join(results1, f'out-{role}.bin'), np.uint8)
    if a.size != n or b.size != n:
        return None
    index = np.arange(n)
    unchanged = ((a == (SENTINEL(index, 0) & 0xFF)) &
                 (b == (SENTINEL(index, 1) & 0xFF)))
    written = ~unchanged
    return {'written': int(written.sum()), 'expected': n,
            'first_unwritten': int(np.argmax(unchanged)) if unchanged.any() else None}


def check(fixture, probe, shaders, results, json_out):
    record = json.load(open(os.path.join(fixture, 'fixture.json')))
    rows, problems = compare_outputs(fixture, results)
    # written-region test: two sentinel seeds
    dir0 = os.path.join(fixture, 'results-seed0')
    dir1 = os.path.join(fixture, 'results-seed1')
    run(fixture, probe, shaders, 0, dir0)
    run(fixture, probe, shaders, 1, dir1)
    for role, entry in record['outputs'].items():
        region = written_region(fixture, dir0, dir1, role, entry)
        if region is None:
            problems.append(f'{role}: written-region run missing')
            continue
        if region['written'] != region['expected']:
            problems.append(f'{role}: only {region["written"]} of {region["expected"]} B '
                            f'were written (first miss at {region["first_unwritten"]})')
        rows.append({'role': f'{role}.written', **region})
        print(f'{role:6s} written region {region["written"]}/{region["expected"]} B')

    expect = record.get('expect') or {}
    for row in rows:
        if 'fraction' in row and row['fraction'] < expect.get('identical_min', 0.0):
            problems.append(f'{row["role"]}: {100 * row["fraction"]:.4f}% identical, '
                            f'below {100 * expect["identical_min"]:.4f}%')
    result = {'fixture': fixture, 'layer': record['layer'], 'rows': rows,
              'problems': problems, 'pass': not problems}
    if json_out:
        json.dump(result, json_out, indent=2)
    print(f'{len(problems)} problem(s)')
    for problem in problems:
        print(f'  {problem}')
    return 0 if not problems else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export')
    e.add_argument('capture')
    e.add_argument('--layer', required=True)
    e.add_argument('--out', required=True)
    e.add_argument('--frame', type=int, default=1)
    r = sub.add_parser('run')
    r.add_argument('fixture')
    r.add_argument('--probe',
                   default=os.path.join('build', 'native', 'Release', 'm_network_stage.exe'))
    r.add_argument('--shaders', default=os.path.join(REPO, 'build', 'native'))
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--out', default=None)
    c = sub.add_parser('check')
    c.add_argument('fixture')
    c.add_argument('--results', default=None)
    c.add_argument('--probe',
                   default=os.path.join('build', 'native', 'Release', 'm_network_stage.exe'))
    c.add_argument('--shaders', default=os.path.join(REPO, 'build', 'native'))
    c.add_argument('--json', type=argparse.FileType('w'), default=None)
    a = ap.parse_args()
    if a.command == 'export':
        return export(a.capture, a.layer, a.out, a.frame)
    if a.command == 'run':
        results = a.out or os.path.join(a.fixture, 'results')
        run(a.fixture, a.probe, a.shaders, a.seed, results)
        return 0
    return check(a.fixture, a.probe, a.shaders,
                 a.results or os.path.join(a.fixture, 'results'), a.json)


if __name__ == '__main__':
    sys.exit(main())
