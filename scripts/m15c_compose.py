#!/usr/bin/env python3
"""M15-C: compose the accepted five-stage frame into one persistent manifest.

The fixture runners execute each stage in its own process and hand tensors over
through files.  This script expresses the same pipeline as one flat list of
dispatches and resources so the persistent engine can keep every device object
alive across frames:

    enc0 INT8 (features -> quantizer -> up -> epilogue)
    ten-layer INT8 chain (fused MLP GELU/quantizer)
    dec0 INT8 (quantizer -> matmul -> epilogue)
    post (FP16 image stage)
    downsample (FP16 image stage)

Handoffs are wired by resource name.  The dec0->post gate/mix surfaces change
their D3D12 view format (R32_UINT producer, R8G8B8A8_UNORM consumer), so they
use an explicit GPU byte-copy; D3D12 cannot alias the two typed views.

  m15c_compose.py --out DIR [--layers DIR] [--fixtures DIR] [--control-mlp]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
TEMP = Path(os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))) / 'Temp/opencode'
DEFAULT_LAYERS = TEMP / 'm09h-complete-chain-20261030/layers'
DEFAULT_FIXTURES = Path(os.environ.get('SETTE_FIXTURES', TEMP / 'm-capture-run/fixtures'))

import m09_complete as complete  # noqa: E402
import m_layer_suite as suite  # noqa: E402
import m_stage_fixture as sf  # noqa: E402


def parse_stage(path):
    """One version-2 image stage: shader, geometry, params and bindings."""
    stage = {'shader': None, 'threads': [1, 1, 1], 'groups': [1, 1, 1],
             'params': None, 'bindings': []}
    for line in Path(path).read_text().splitlines():
        words = line.split()
        if not words or words[0].startswith('#'):
            continue
        key = words[0]
        if key == 'version':
            continue
        if key == 'shader':
            stage['shader'] = words[1]
        elif key == 'threads':
            stage['threads'] = [int(v) for v in words[1:4]]
        elif key == 'groups':
            stage['groups'] = [int(v) for v in words[1:4]]
        elif key == 'params':
            stage['params'] = (words[1], int(words[2]))
        elif key in ('srv', 'uav'):
            binding = {'input': key == 'srv', 'reg': int(words[2])}
            if words[1] == 'tex2d':
                binding.update(kind='tex2d', format=words[3], width=int(words[4]),
                               height=int(words[5]), file=words[6])
            elif words[1] == 'buffer':
                binding.update(kind='buffer', file=words[3], bytes=int(words[4]))
            else:
                raise SystemExit(f'{path}: unknown binding kind {words[1]}')
            stage['bindings'].append(binding)
        elif key in ('enc0_int8', 'dec0_int8'):
            continue
        else:
            raise SystemExit(f'{path}: unknown key {key}')
    return stage


class Manifest:
    def __init__(self):
        self.lines = ['version 3']
        self.resources = {}
        self.stages = []
        self.dumps = []

    def resource(self, name, kind, **fields):
        key = (kind, fields.get('format'), fields.get('width'), fields.get('height'),
               fields.get('bytes'), fields.get('init'), fields.get('external'),
               fields.get('committed'))
        if name in self.resources:
            if self.resources[name] != key:
                raise SystemExit(f'conflicting declarations for {name}')
            return
        self.resources[name] = key
        if kind == 'buffer':
            line = f'resource {name} buffer {fields["bytes"]}'
        else:
            line = (f'resource {name} tex2d {fields["format"]} '
                    f'{fields["width"]} {fields["height"]}')
        if fields.get('external'):
            line += ' external'
        elif fields.get('committed'):
            line += ' committed'
        elif fields.get('init'):
            line += f' {fields["init"]}'
        self.lines.append(line)

    def stage(self, name, shader, threads, groups, params, bindings):
        body = [f'stage {name}', f'shader {shader}',
                f'threads {threads[0]} {threads[1]} {threads[2]}',
                f'groups {groups[0]} {groups[1]} {groups[2]}',
                f'params {params[0]} {params[1]}']
        for b in bindings:
            if b['input']:
                body.append(f'srv {b["kind"]} {b["reg"]} {b["resource"]}')
            else:
                body.append(f'uav {b["kind"]} {b["reg"]} {b["resource"]}')
        self.stages.extend(body)

    def copy(self, name, src, dst):
        self.stages.extend([f'stage {name}', f'copy tex2d {src} {dst}'])

    def dump(self, name, filename):
        self.dumps.append(f'dump {name} {filename}')

    def write(self, path):
        text = '\n'.join(self.lines + self.stages + self.dumps) + '\n'
        Path(path).write_text(text)


def bind(binding, resource):
    out = dict(binding)
    out['resource'] = resource
    return out


def normalize(bindings):
    """Sequential SRV/UAV registers, as the engine's root signature expects."""
    srv = uav = 0
    for b in bindings:
        b['reg'] = srv if b['input'] else uav
        if b['input']:
            srv += 1
        else:
            uav += 1
    return bindings


def compose(out, layers, fixtures, fused=True):
    out = Path(out)
    work = out / 'work'
    work.mkdir(parents=True, exist_ok=True)

    def fresh(name):
        target = work / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(fixtures / name, target,
                        ignore=shutil.ignore_patterns('results*', 'dbg*', 'd[0-9]*',
                                                      'ref-*.bin', '*.log'))
        return target

    enc0 = fresh('enc0')
    dec0 = fresh('dec0')
    post = fresh('post')
    down = fresh('downsample')
    complete.prepare_enc0(enc0)
    sf.prepare_dec0_mode(dec0, 'int8-expansion')

    chain = work / 'chain'
    if chain.exists():
        shutil.rmtree(chain)
    suite.build_chain(str(layers), str(chain), fuse_mlp=fused)
    chain_graph = chain / 'graph.txt'

    # ------------------------------------------------------------- resources
    m = Manifest()
    # External colour input (enc0 encoder path) and caller-owned output.
    m.resource('enc0_in_colour', 'tex2d', format='R16G16B16A16_FLOAT', width=960,
               height=540, bytes=960 * 540 * 8, external=True)
    m.resource('frame_out', 'tex2d', format='R16G16B16A16_FLOAT', width=1920,
               height=1080, bytes=1920 * 1080 * 8, external=True)

    m.resource('enc0_weights', 'buffer', bytes=12416, init=str(enc0 / 'in-weights.bin'))
    m.resource('enc0_features', 'buffer', bytes=33423360)
    m.resource('enc0_q8a', 'buffer', bytes=16711680)
    m.resource('enc0_q8as', 'buffer', bytes=522240)
    m.resource('enc0_embedding', 'buffer', bytes=16711680)
    m.resource('enc0_acc', 'buffer', bytes=33423360)
    m.resource('enc0_float', 'buffer', bytes=33423360)
    m.resource('enc0_wc', 'buffer', bytes=8192, init=str(enc0 / 'enc0-wc.bin'))
    m.resource('enc0_ws', 'buffer', bytes=256, init=str(enc0 / 'enc0-ws.bin'))
    m.resource('enc0_bias', 'buffer', bytes=256, init=str(enc0 / 'enc0-bias.bin'))
    shape = work / 'enc0-shape.bin'
    shape.write_bytes((130560).to_bytes(4, 'little') + (128).to_bytes(4, 'little') +
                      (64).to_bytes(4, 'little') + (0).to_bytes(4, 'little'))
    m.resource('enc0_shape', 'buffer', bytes=16, init=str(shape))
    m.resource('enc0_tokens', 'buffer', bytes=8355840)

    # Chain resources. enc1_in is produced by enc0; dec1_out is dec0's input.
    _, chain_stages, _ = suite._parse_graph(chain_graph)
    graph_resources = {}
    for line in chain_graph.read_text().splitlines():
        words = line.split()
        if words and words[0] == 'resource':
            graph_resources[words[1]] = {'bytes': int(words[2]),
                                         'init': words[3] if len(words) > 3 else None}
    for name, spec in graph_resources.items():
        if name == 'enc1_in':
            continue
        target = 'dec0_tokens' if name == 'dec1_out' else f'chain_{name}'
        m.resource(target, 'buffer', bytes=spec['bytes'], init=spec['init'])
    # Ensure dec0_tokens exists even when the graph names it differently.
    if 'dec0_tokens' not in m.resources:
        m.resource('dec0_tokens', 'buffer', bytes=8355840)

    # Dec0 prefix and epilogue.
    m.resource('dec0_weights', 'buffer', bytes=8448, init=str(dec0 / 'in-weights.bin'))
    m.resource('dec0_q8a', 'buffer', bytes=8355840)
    m.resource('dec0_q8as', 'buffer', bytes=522240)
    m.resource('dec0_acc', 'buffer', bytes=41779200)
    m.resource('dec0_float', 'buffer', bytes=41779200)
    m.resource('dec0_scratch', 'buffer', bytes=16)
    m.resource('dec0_expansion', 'buffer', bytes=20889600)
    m.resource('dec0_wc', 'buffer', bytes=5120, init=str(dec0 / 'int8-wc.bin'))
    m.resource('dec0_ws', 'buffer', bytes=320, init=str(dec0 / 'int8-ws.bin'))
    m.resource('dec0_bias', 'buffer', bytes=320, init=str(dec0 / 'int8-bias.bin'))
    m.resource('dec0_gate', 'tex2d', format='R32_UINT', width=960, height=540,
               bytes=960 * 540 * 4)
    m.resource('dec0_mix', 'tex2d', format='R32_UINT', width=960, height=540,
               bytes=960 * 540 * 4)
    m.resource('dec0_feat', 'tex2d', format='R32_UINT', width=960, height=540,
               bytes=960 * 540 * 4)
    m.resource('dec0_cov', 'tex2d', format='R16G16B16A16_FLOAT', width=960, height=540,
               bytes=960 * 540 * 8)

    # Post inputs that are static reconstruction surfaces, and its outputs.
    m.resource('post_colour', 'tex2d', format='R16G16B16A16_FLOAT', width=960,
               height=540, bytes=960 * 540 * 8, init=str(post / 'in-colour.bin'))
    m.resource('post_gate', 'tex2d', format='R8G8B8A8_UNORM', width=960, height=540,
               bytes=960 * 540 * 4)
    m.resource('post_mix', 'tex2d', format='R8G8B8A8_UNORM', width=960, height=540,
               bytes=960 * 540 * 4)
    m.resource('post_motion', 'tex2d', format='R16G16_FLOAT', width=960, height=540,
               bytes=960 * 540 * 4, init=str(post / 'in-motion.bin'))
    m.resource('post_packed', 'buffer', bytes=18662400)
    m.resource('post_history', 'tex2d', format='R32_UINT', width=5760, height=1620,
               bytes=5760 * 1620 * 4)
    m.resource('post_half', 'tex2d', format='R32G32B32A32_FLOAT', width=1440, height=810,
               bytes=1440 * 810 * 16)
    m.resource('down_colour', 'tex2d', format='R16G16B16A16_FLOAT', width=960,
               height=540, bytes=960 * 540 * 8, init=str(down / 'in-colour.bin'))

    # --------------------------------------------------------------- stages
    enc0_params = str(enc0 / 'params.bin')
    m.stage('enc0.pf-features', 'enc0_q8_features.cso', [64, 1, 1], [2040, 1, 1],
            (enc0_params, 24),
            [{'input': True, 'kind': 'buffer', 'reg': 0, 'resource': 'enc0_weights'},
             {'input': True, 'kind': 'tex2d', 'reg': 1, 'resource': 'enc0_in_colour'},
             {'input': False, 'kind': 'buffer', 'reg': 0, 'resource': 'enc0_features'}])
    m.stage('enc0.pf-quant', 'swin_q8_quant.cso', [256, 1, 1], [2040, 1, 1],
            (str(shape), 16),
            [{'input': True, 'kind': 'buffer', 'reg': 0, 'resource': 'enc0_features'},
             {'input': False, 'kind': 'buffer', 'reg': 0, 'resource': 'enc0_q8a'},
             {'input': False, 'kind': 'buffer', 'reg': 1, 'resource': 'enc0_q8as'}])
    m.stage('enc0.pf-up', 'swin_mlp_up_q8.cso', [64, 1, 1], [8160, 1, 1],
            (str(shape), 16),
            [{'input': True, 'kind': 'buffer', 'reg': 0, 'resource': 'enc0_q8a'},
             {'input': True, 'kind': 'buffer', 'reg': 1, 'resource': 'enc0_wc'},
             {'input': True, 'kind': 'buffer', 'reg': 2, 'resource': 'enc0_q8as'},
             {'input': True, 'kind': 'buffer', 'reg': 3, 'resource': 'enc0_ws'},
             {'input': True, 'kind': 'buffer', 'reg': 4, 'resource': 'enc0_bias'},
             {'input': False, 'kind': 'buffer', 'reg': 0, 'resource': 'enc0_embedding'},
             {'input': False, 'kind': 'buffer', 'reg': 1, 'resource': 'enc0_acc'},
             {'input': False, 'kind': 'buffer', 'reg': 2, 'resource': 'enc0_float'}])
    enc0_epilogue = parse_stage(enc0 / 'stage.txt')
    ep_inputs, ep_outputs = [], []
    for b in enc0_epilogue['bindings']:
        if b['kind'] == 'buffer' and b['file'] == 'in-weights.bin':
            ep_inputs.append(bind(b, 'enc0_weights'))
        elif b['kind'] == 'tex2d':
            ep_inputs.append(bind(b, 'enc0_in_colour'))
        elif b['kind'] == 'buffer' and b['file'] == 'out-tokens.bin':
            ep_outputs.append(bind(b, 'enc0_tokens'))
        else:
            raise SystemExit(f'unexpected enc0 epilogue binding {b}')
    ep_inputs.append({'input': True, 'kind': 'buffer', 'resource': 'enc0_embedding'})
    m.stage('enc0.epilogue', enc0_epilogue['shader'], enc0_epilogue['threads'],
            enc0_epilogue['groups'],
            (str(enc0 / enc0_epilogue['params'][0]), enc0_epilogue['params'][1]),
            normalize(ep_inputs + ep_outputs))

    for stage in chain_stages:
        lines = {w[0]: w for w in stage['lines']}
        shader = Path(lines['shader'][1]).name
        threads = [int(v) for v in lines['threads'][1:4]]
        groups = [int(v) for v in lines['groups'][1:4]]
        params = (lines['params'][1], int(lines['params'][2]))
        bindings = []
        for words in stage['lines']:
            if words[0] not in ('srv', 'uav'):
                continue
            resource = words[2]
            if resource == 'enc1_in':
                resource = 'enc0_tokens'
            elif resource == 'dec1_out':
                resource = 'dec0_tokens'
            elif not resource.startswith('chain_'):
                graph = [n for n in graph_resources if n == resource]
                resource = f'chain_{resource}' if graph else resource
            bindings.append({'input': words[0] == 'srv', 'kind': 'buffer',
                             'reg': int(words[1]), 'resource': resource})
        m.stage(stage['name'], shader, threads, groups, params, bindings)

    dec0_params = str(dec0 / 'params.bin')
    m.stage('dec0.pf-quant', 'dec0_q8_quant.cso', [64, 1, 1], [2040, 1, 1],
            (dec0_params, 32),
            [{'input': True, 'kind': 'buffer', 'reg': 0, 'resource': 'dec0_weights'},
             {'input': True, 'kind': 'buffer', 'reg': 1, 'resource': 'dec0_tokens'},
             {'input': False, 'kind': 'buffer', 'reg': 0, 'resource': 'dec0_scratch'},
             {'input': False, 'kind': 'buffer', 'reg': 1, 'resource': 'dec0_scratch'},
             {'input': False, 'kind': 'buffer', 'reg': 2, 'resource': 'dec0_scratch'},
             {'input': False, 'kind': 'buffer', 'reg': 3, 'resource': 'dec0_scratch'},
             {'input': False, 'kind': 'buffer', 'reg': 4, 'resource': 'dec0_q8a'},
             {'input': False, 'kind': 'buffer', 'reg': 5, 'resource': 'dec0_q8as'}])
    m.stage('dec0.pf-matmul', 'dec0_q8_matmul.cso', [64, 1, 1], [2040, 40, 1],
            (dec0_params, 32),
            [{'input': True, 'kind': 'buffer', 'reg': 0, 'resource': 'dec0_q8a'},
             {'input': True, 'kind': 'buffer', 'reg': 1, 'resource': 'dec0_wc'},
             {'input': True, 'kind': 'buffer', 'reg': 2, 'resource': 'dec0_q8as'},
             {'input': True, 'kind': 'buffer', 'reg': 3, 'resource': 'dec0_ws'},
             {'input': True, 'kind': 'buffer', 'reg': 4, 'resource': 'dec0_bias'},
             {'input': False, 'kind': 'buffer', 'reg': 0, 'resource': 'dec0_expansion'},
             {'input': False, 'kind': 'buffer', 'reg': 1, 'resource': 'dec0_acc'},
             {'input': False, 'kind': 'buffer', 'reg': 2, 'resource': 'dec0_float'}])
    dec0_epilogue = parse_stage(dec0 / 'stage.txt')
    de_inputs, de_outputs = [], []
    for b in dec0_epilogue['bindings']:
        if b['kind'] == 'buffer' and b['file'] == 'in-weights.bin':
            de_inputs.append(bind(b, 'dec0_weights'))
        elif b['kind'] == 'buffer' and b['file'] == 'in-tokens.bin':
            de_inputs.append(bind(b, 'dec0_tokens'))
        elif b['kind'] == 'tex2d':
            role = Path(b['file']).stem.replace('out-', '')
            de_outputs.append(bind(b, f'dec0_{role}'))
        else:
            raise SystemExit(f'unexpected dec0 epilogue binding {b}')
    de_inputs.append({'input': True, 'kind': 'buffer', 'resource': 'dec0_expansion'})
    m.stage('dec0.epilogue', dec0_epilogue['shader'], dec0_epilogue['threads'],
            dec0_epilogue['groups'],
            (str(dec0 / dec0_epilogue['params'][0]), dec0_epilogue['params'][1]),
            normalize(de_inputs + de_outputs))

    m.copy('copy.gate', 'dec0_gate', 'post_gate')
    m.copy('copy.mix', 'dec0_mix', 'post_mix')

    post_stage = parse_stage(post / 'stage.txt')
    post_bindings = []
    for b in post_stage['bindings']:
        if b['kind'] == 'buffer':
            post_bindings.append(bind(b, 'post_packed'))
        else:
            role = Path(b['file']).stem.replace('in-', '').replace('out-', '')
            if b['input']:
                if role == 'cov':
                    post_bindings.append(bind(b, 'dec0_cov'))
                elif role == 'gate':
                    post_bindings.append(bind(b, 'post_gate'))
                elif role == 'mix':
                    post_bindings.append(bind(b, 'post_mix'))
                elif role == 'colour':
                    post_bindings.append(bind(b, 'post_colour'))
                elif role == 'motion':
                    post_bindings.append(bind(b, 'post_motion'))
                else:
                    raise SystemExit(f'unexpected post input {b}')
            else:
                post_bindings.append(bind(b, f'post_{role}'))
    m.stage('post.main', post_stage['shader'], post_stage['threads'], post_stage['groups'],
            (str(post / post_stage['params'][0]), post_stage['params'][1]),
            normalize(post_bindings))

    down_stage = parse_stage(down / 'stage.txt')
    down_bindings = []
    for b in down_stage['bindings']:
        if b['kind'] == 'buffer':
            down_bindings.append(bind(b, 'post_packed'))
        elif b['input'] and Path(b['file']).stem == 'in-colour':
            down_bindings.append(bind(b, 'down_colour'))
        elif not b['input']:
            down_bindings.append(bind(b, 'frame_out'))
        else:
            raise SystemExit(f'unexpected downsample binding {b}')
    m.stage('down.main', down_stage['shader'], down_stage['threads'],
            down_stage['groups'],
            (str(down / down_stage['params'][0]), down_stage['params'][1]),
            normalize(down_bindings))

    # --------------------------------------------------------------- dumps
    m.dump('enc0_tokens', 'out-enc0-tokens.bin')
    m.dump('dec0_tokens', 'out-dec1.bin')
    for name in ('enc1_merge', 'enc1_out', 'enc2_merge', 'enc2_out',
                 'tube_1_out', 'tube_2_out', 'tube_3_out', 'tube_4_out',
                 'tube_5_out', 'tube_6_out', 'dec2_out'):
        if f'chain_{name}' in m.resources:
            m.dump(f'chain_{name}', f'out-{name.replace("_", "-")}.bin')
    m.dump('dec0_gate', 'out-dec0-gate.bin')
    m.dump('dec0_mix', 'out-dec0-mix.bin')
    m.dump('dec0_feat', 'out-dec0-feat.bin')
    m.dump('dec0_cov', 'out-dec0-cov.bin')
    m.dump('post_packed', 'out-post-packed.bin')
    m.dump('post_history', 'out-post-history.bin')
    m.dump('post_half', 'out-post-half.bin')
    m.dump('frame_out', 'out-frame.bin')

    manifest = out / 'engine.txt'
    m.write(manifest)
    summary = {'manifest': str(manifest), 'stages': len(
        [line for line in m.stages if line.startswith('stage ')]),
        'resources': len(m.resources), 'dumps': len(m.dumps),
        'fused_mlp': fused, 'work': str(work)}
    (out / 'compose.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--layers', type=Path, default=DEFAULT_LAYERS)
    ap.add_argument('--fixtures', type=Path, default=DEFAULT_FIXTURES)
    ap.add_argument('--control-mlp', action='store_true',
                    help='use the accepted unfused GELU/quantizer pair')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; use a fresh directory')
    a.out.mkdir(parents=True)
    if not (a.layers / 'dec1' / 'graph.txt').is_file():
        raise SystemExit(f'missing layer fixtures: {a.layers}')
    for name in ('enc0', 'dec0', 'post', 'downsample'):
        if not (a.fixtures / name / 'stage.txt').is_file():
            raise SystemExit(f'missing fixture: {a.fixtures / name}')
    compose(a.out, a.layers, a.fixtures, fused=not a.control_mlp)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
