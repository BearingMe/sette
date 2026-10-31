"""M15-C/M16: persistent engine composition, external resources and lifecycle.

One composed tree is shared by the whole module and removed at teardown, because
composing copies the accepted layer fixtures.  GPU tests are skipped when the
fixtures or the built probe are absent.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
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


def canonical():
    return {
        'enc0_in_colour': str(FIXTURES / 'enc0' / 'in-colour.bin'),
        'post_colour': str(FIXTURES / 'post' / 'in-colour.bin'),
        'post_motion': str(FIXTURES / 'post' / 'in-motion.bin'),
        'down_colour': str(FIXTURES / 'downsample' / 'in-colour.bin'),
    }


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
    root = tmp_path_factory.mktemp('m16-compose')
    try:
        out = root / 'compose'
        compose.compose(out, LAYERS, FIXTURES, fused=True)
        yield out / 'engine.txt'
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run(command):
    done = subprocess.run([str(c) for c in command], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    return done


def write_frames(path, frames):
    Path(path).write_text(json.dumps(frames, indent=2), encoding='utf-8')
    return path


def engine(composed, frames_path, out, *extra):
    out = Path(out)
    command = [PROBE, '--manifest', composed, '--shaders', SHADERS,
               '--output', 'frame_out', '--frames-file', frames_path,
               '--report', out / 'report.json', *extra]
    run(command)
    return json.loads((out / 'report.json').read_text())


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
    for name in ('enc0_in_colour', 'post_colour', 'post_motion', 'down_colour',
                 'frame_out'):
        assert resources[name][-1] == 'external'
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
    frames = write_frames(tmp_path / 'frames.json', [canonical()])
    out = tmp_path / 'run'
    report = engine(composed, frames, out, '--mode', 'validation',
                    '--dump-dir', out / 'dumps', '--dump-frame', '0')
    assert sha(out / 'dumps' / 'frame-00' / 'out-frame.bin') == ACCEPTED_FRAME
    assert sha(out / 'dumps' / 'frame-00' / 'out-dec1.bin') == ACCEPTED_CHAIN
    assert report['runs'][0]['dispatches'] == 233
    assert report['runs'][0]['copies'] == 2
    assert report['runs'][0]['external_rebinds'] == 6
    assert report['memory']['arena_heap_bytes'] > 0
    assert report['memory']['external_bytes'] == 31104000


@pytest.mark.skipif(not PROBE.is_file(), reason='persistent engine is not built')
def test_caller_owned_command_list_reproduces_the_accepted_frame(composed, tmp_path):
    require_probe()
    frames = write_frames(tmp_path / 'frames.json', [canonical()])
    out = tmp_path / 'caller'
    report = engine(composed, frames, out, '--caller-list',
                    '--caller-readback', out / 'frame.bin')
    assert sha(out / 'frame.bin') == ACCEPTED_FRAME
    row = report['runs'][0]
    assert row['gpu_valid'] is False
    assert row['engine_submitted'] is False
    assert row['engine_waited'] is False
    assert row['record_ms'] > 0


@pytest.mark.skipif(not PROBE.is_file(), reason='persistent engine is not built')
def test_external_device_and_queue_reproduce_the_accepted_frame(composed, tmp_path):
    require_probe()
    frames = write_frames(tmp_path / 'frames.json', [canonical()])
    out = tmp_path / 'external'
    report = engine(composed, frames, out, '--external-device', '--caller-list',
                    '--caller-readback', out / 'frame.bin')
    assert sha(out / 'frame.bin') == ACCEPTED_FRAME
    assert report['recording'] == 'caller-owned'
    assert report['runs'][0]['engine_submitted'] is False


@pytest.mark.skipif(not PROBE.is_file(), reason='persistent engine is not built')
def test_persistent_engine_reuses_resources_across_frames(composed, tmp_path):
    require_probe()
    frame = canonical()
    frames = write_frames(tmp_path / 'frames.json', [frame, frame])
    out = tmp_path / 'two-frames'
    report = engine(composed, frames, out)
    assert len(report['runs']) == 2
    assert report['init']['stages'] == 235
    assert report['memory']['arena_heap_bytes'] > 0
    assert all(row['dispatches'] == 233 for row in report['runs'])
    # Each frame owns its uploaded input textures, so frame 1 rewrites the
    # five input descriptors (enc0 binds colour twice); the output is shared.
    assert report['runs'][0]['external_rebinds'] == 6
    assert report['runs'][1]['external_rebinds'] == 5


@pytest.mark.skipif(not PROBE.is_file(), reason='persistent engine is not built')
def test_motion_is_a_validated_but_unread_reset_input(composed, tmp_path):
    require_probe()
    frame = canonical()
    motion = np.zeros((540, 960, 2), np.float16)
    motion_path = tmp_path / 'motion-zero.bin'
    motion_path.write_bytes(motion.tobytes())
    changed = dict(frame, post_motion=str(motion_path))
    frames = write_frames(tmp_path / 'frames.json', [frame, changed])
    out = tmp_path / 'motion'
    engine(composed, frames, out, '--mode', 'validation',
           '--dump-dir', out / 'dumps', '--dump-frame', '0')
    first = sha(out / 'dumps' / 'frame-00' / 'out-frame.bin')
    second = sha(out / 'dumps' / 'frame-01' / 'out-frame.bin')
    assert first == second == ACCEPTED_FRAME


@pytest.mark.skipif(not PROBE.is_file(), reason='persistent engine is not built')
def test_post_colour_is_a_live_input(composed, tmp_path):
    require_probe()
    colour = np.fromfile(FIXTURES / 'post' / 'in-colour.bin', '<f2')
    colour = colour.reshape(540, 960, 4)
    shifted = colour.copy()
    shifted[..., :3] = np.roll(colour[..., :3], 16, axis=1)
    shifted_path = tmp_path / 'post-shifted.bin'
    shifted_path.write_bytes(shifted.tobytes())
    frames = write_frames(tmp_path / 'frames.json',
                          [dict(canonical(), post_colour=str(shifted_path))])
    out = tmp_path / 'post'
    engine(composed, frames, out, '--mode', 'validation',
           '--dump-dir', out / 'dumps', '--dump-frame', '0')
    assert sha(out / 'dumps' / 'frame-00' / 'out-frame.bin') != ACCEPTED_FRAME


@pytest.mark.skipif(not PROBE.is_file(), reason='persistent engine is not built')
def test_downsample_colour_alpha_is_a_live_input(composed, tmp_path):
    require_probe()
    colour = np.fromfile(FIXTURES / 'downsample' / 'in-colour.bin', '<f2')
    colour = colour.reshape(540, 960, 4)
    changed = colour.copy()
    changed[..., 3] = np.where(colour[..., 3] > np.float16(0.5),
                               np.float16(0.0), np.float16(1.0))
    changed_path = tmp_path / 'down-alpha.bin'
    changed_path.write_bytes(changed.tobytes())
    frames = write_frames(tmp_path / 'frames.json',
                          [dict(canonical(), down_colour=str(changed_path))])
    out = tmp_path / 'down'
    engine(composed, frames, out, '--mode', 'validation',
           '--dump-dir', out / 'dumps', '--dump-frame', '0')
    assert sha(out / 'dumps' / 'frame-00' / 'out-frame.bin') != ACCEPTED_FRAME


@pytest.mark.skipif(not PROBE.is_file(), reason='persistent engine is not built')
def test_external_selftest_rejects_invalid_resources(composed, tmp_path):
    require_probe()
    out = tmp_path / 'selftest'
    command = [PROBE, '--manifest', composed, '--shaders', SHADERS,
               '--output', 'frame_out', '--external-selftest', out,
               '--report', out / 'report.json', '--debug']
    for name, path in canonical().items():
        command += ['--input', f'{name}={path}']
    run(command)
    report = json.loads((out / 'external-selftest.json').read_text())
    assert report['status'] == 'PASS'
    assert len(report['cases']) == 16
    assert all(case['expected'] == case['observed'] for case in report['cases'])
    assert report['cases'][-1]['name'] == 'zero_debug_errors'


@pytest.mark.skipif(not PROBE.is_file(), reason='persistent engine is not built')
def test_cancelled_borrowed_frame_is_never_submitted(composed, tmp_path):
    require_probe()
    frames = write_frames(tmp_path / 'frames.json', [canonical()])
    out = tmp_path / 'cancel'
    report = engine(composed, frames, out, '--caller-list', '--caller-cancel')
    assert report['runs'][0]['dispatches'] == 233
    assert report['runs'][0]['engine_submitted'] is False
    assert report['runs'][0]['engine_waited'] is False
