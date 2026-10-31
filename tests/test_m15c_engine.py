"""M15-C: persistent engine composition, lifecycle and external resources.

The CPU test locks the composed pipeline shape.  The GPU tests execute the real
engine and require the accepted fixtures and a built `m_inference_stage`; they
are skipped when either is absent.  One composed tree is shared by the whole
module and removed at teardown, because composing copies the accepted layer
fixtures.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'scripts'))
import m15c_compose as compose  # noqa: E402

TEMP = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'Temp/opencode'
LAYERS = TEMP / 'm09h-complete-chain-20261030/layers'
FIXTURES = Path(os.environ.get('SETTE_FIXTURES', TEMP / 'm-capture-run/fixtures'))
PROBE = HERE / 'build/native/Release/m_inference_stage.exe'
SHADERS = HERE / 'build/native'

ACCEPTED_FRAME = 'f76ac032f45a91dc6c4ead22979eff16ed1d04c4a4cf1add2ff1adc0bd4d6c00'
ACCEPTED_CHAIN = 'c89208ed744acd71275f349d99d758c16bd55c77a6fb5d3da88109a3573c6b0d'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_fixtures():
    if not (LAYERS / 'dec1' / 'graph.txt').is_file():
        pytest.skip('accepted layer fixtures are not present')
    for name in ('enc0', 'dec0', 'post', 'downsample'):
        if not (FIXTURES / name / 'stage.txt').is_file():
            pytest.skip(f'accepted {name} fixture is not present')


def require_probe():
    if not PROBE.is_file() or not (SHADERS / 'swin_mlp_gelu_quant.cso').is_file():
        pytest.skip('the persistent-engine probe is not built')


@pytest.fixture(scope='module')
def composed(tmp_path_factory):
    require_fixtures()
    root = tmp_path_factory.mktemp('m15c-compose')
    try:
        out = root / 'compose'
        compose.compose(out, LAYERS, FIXTURES, fused=True)
        yield out / 'engine.txt'
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run(command):
    done = subprocess.run(command, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    return done


def manifest_lines(path):
    return [line.split() for line in Path(path).read_text().splitlines()
            if line.strip() and not line.startswith('#')]


def test_composed_manifest_matches_the_accepted_frame(composed):
    lines = manifest_lines(composed)
    stages = [w[1] for w in lines if w[0] == 'stage']
    resources = {w[1]: w for w in lines if w[0] == 'resource'}
    dumps = [w for w in lines if w[0] == 'dump']
    copies = [w for w in lines if w[0] == 'copy']
    assert len(stages) == 235
    assert len(resources) == 715
    assert len(copies) == 2
    assert len(dumps) == 21
    assert stages[:4] == ['enc0.pf-features', 'enc0.pf-quant', 'enc0.pf-up',
                          'enc0.epilogue']
    assert sum(name.endswith('.mlp-fused') for name in stages) == 10
    assert stages[-5:] == ['dec0.epilogue', 'copy.gate', 'copy.mix',
                           'post.main', 'down.main']
    assert resources['enc0_in_colour'][-1] == 'external'
    assert resources['frame_out'][-1] == 'external'
    # Handoff tensors are declared once and shared by producer and consumer.
    assert any(w[0] == 'uav' and w[-1] == 'enc0_tokens' for w in lines)
    assert any(w[0] == 'srv' and w[-1] == 'enc0_tokens' for w in lines)
    assert any(w[0] == 'uav' and w[-1] == 'dec0_tokens' for w in lines)
    assert any(w[0] == 'srv' and w[-1] == 'dec0_tokens' for w in lines)
    assert any(w[0] == 'uav' and w[-1] == 'post_packed' for w in lines)
    assert any(w[0] == 'srv' and w[-1] == 'post_packed' for w in lines)


@pytest.mark.skipif(not PROBE.is_file(), reason='persistent engine is not built')
def test_persistent_engine_reproduces_the_accepted_frame(composed, tmp_path):
    require_probe()
    colour = FIXTURES / 'enc0' / 'in-colour.bin'
    out = tmp_path / 'run'
    run([str(PROBE), '--manifest', str(composed), '--shaders', str(SHADERS),
         '--input', f'enc0_in_colour={colour}', '--output', 'frame_out',
         '--frames', '1', '--mode', 'validation', '--dump-dir', str(out / 'dumps'),
         '--dump-frame', '0', '--report', str(out / 'report.json')])
    assert sha(out / 'dumps' / 'frame-00' / 'out-frame.bin') == ACCEPTED_FRAME
    assert sha(out / 'dumps' / 'frame-00' / 'out-dec1.bin') == ACCEPTED_CHAIN
    report = json.loads((out / 'report.json').read_text())
    assert report['runs'][0]['dispatches'] == 233
    assert report['runs'][0]['copies'] == 2
    assert report['memory']['arena_heap_bytes'] > 0
    assert report['memory']['placed_allocations'] > 0


@pytest.mark.skipif(not PROBE.is_file(), reason='persistent engine is not built')
def test_caller_owned_command_list_reproduces_the_accepted_frame(composed, tmp_path):
    require_probe()
    colour = FIXTURES / 'enc0' / 'in-colour.bin'
    out = tmp_path / 'caller'
    run([str(PROBE), '--manifest', str(composed), '--shaders', str(SHADERS),
         '--input', f'enc0_in_colour={colour}', '--output', 'frame_out',
         '--frames', '1', '--caller-list', '--caller-readback', str(out / 'frame.bin'),
         '--report', str(out / 'report.json')])
    assert sha(out / 'frame.bin') == ACCEPTED_FRAME
    report = json.loads((out / 'report.json').read_text())
    # A borrowed list is recorded but never submitted by the engine, so it has
    # no engine-owned GPU interval; the CPU record time is still reported.
    assert report['runs'][0]['gpu_valid'] is False
    assert report['runs'][0]['record_ms'] > 0


@pytest.mark.skipif(not PROBE.is_file(), reason='persistent engine is not built')
def test_persistent_engine_reuses_resources_across_frames(composed, tmp_path):
    require_probe()
    colour = FIXTURES / 'enc0' / 'in-colour.bin'
    out = tmp_path / 'two-frames'
    run([str(PROBE), '--manifest', str(composed), '--shaders', str(SHADERS),
         '--input', f'enc0_in_colour={colour}', '--input', f'enc0_in_colour={colour}',
         '--output', 'frame_out', '--frames', '2',
         '--report', str(out / 'report.json')])
    report = json.loads((out / 'report.json').read_text())
    assert len(report['runs']) == 2
    assert report['init']['stages'] == 235
    assert report['memory']['arena_heap_bytes'] > 0
    assert all(row['dispatches'] == 233 for row in report['runs'])
    # The allocation inventory is the engine's one-time init, not a per-frame
    # count, so it is identical for a one- and a two-frame run.
    assert report['memory']['committed_allocations'] > 0
    assert report['memory']['placed_allocations'] > 0
