"""M10 temporal-sequence contract.

The sequence driver is built from the reset-only contract, so these tests fix
that contract rather than a captured frame: deterministic frame content, the
persistent-state schema, the SNORM8 decode, and the exact flags each mode
forwards to the approved B6 harness.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('PIL')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import m10_temporal as m10

SHAPE = (540, 960, 4)


def colour(seed=7):
    rng = np.random.default_rng(seed)
    data = rng.random(SHAPE).astype(np.float16)
    data[..., 3] = np.float16(1.0)
    return data


def test_state_schema_matches_declared_byte_counts():
    assert m10.STATE
    for name, spec in m10.STATE.items():
        size = np.dtype(spec['dtype']).itemsize * int(np.prod(spec['shape']))
        assert size == spec['bytes'], name


def test_scenarios_are_deterministic_alpha_preserving_and_in_range():
    source = colour()
    for name, _ in m10.SCENARIOS:
        a, b = m10.variant(source, name), m10.variant(source, name)
        assert a.dtype == np.float16 and a.shape == SHAPE, name
        assert np.array_equal(a, b), name
        assert np.array_equal(a[..., 3], source[..., 3]), name
        assert np.isfinite(a).all(), name
        assert float(a[..., :3].min()) >= 0.0 and float(a[..., :3].max()) <= 1.0, name


def test_declared_scenarios_change_content():
    source = colour()
    base = m10.variant(source, 'static')
    for name in ('pan-8', 'pan-32v', 'rotate-3', 'fast-96', 'disocclude', 'repeat',
                 'thin', 'abrupt'):
        assert not np.array_equal(m10.variant(source, name), base), name


def test_hold_recover_and_final_frame_are_the_canonical_content():
    source = colour()
    base = m10.variant(source, 'static')
    for name in ('hold', 'recover', 'static-end'):
        assert np.array_equal(m10.variant(source, name), base), name


def test_unknown_scenario_is_rejected():
    with pytest.raises(SystemExit):
        m10.variant(colour(), 'not-a-scenario')


def test_describe_identical_and_one_ulp():
    a = np.array([[1.0, 2.0]], np.float16)
    row = m10.describe(a, a.copy())
    assert row['identical'] is True and row['differing_elements'] == 0
    assert row['rms'] == 0.0 and row['max_abs'] == 0.0 and row['non_finite'] == 0

    b = np.nextafter(a, np.float16(4.0))
    row = m10.describe(a, b)
    assert row['identical'] is False and row['differing_elements'] == 2
    assert row['max_abs'] == pytest.approx(float(b[0, 1]) - float(a[0, 1]))


def test_describe_snorm8_uses_the_dec0_decode():
    plus = np.array([127], np.uint8)   # +1.0
    minus = np.array([129], np.uint8)  # -1.0
    row = m10.describe(plus, minus)
    assert row['unit'] == 'snorm8'
    assert row['quantum'] == pytest.approx(1.0 / 127.0)
    assert row['max_abs'] == pytest.approx(2.0)
    # -128 is never produced by the encoder; if it appears it decodes to -1.
    assert m10.describe(np.array([128], np.uint8), np.array([129], np.uint8))['max_abs'] == 0.0


def test_int8_mode_forwards_the_accepted_cumulative_flags():
    flags = m10.MODES['int8']
    assert flags[flags.index('--cumulative-mode') + 1] == 'spatial'
    assert flags[flags.index('--attention-mode') + 1] == 'int8'
    assert flags[flags.index('--enc0-mode') + 1] == 'int8-embedding'
    # `dec0` is set by cumulative mode; passing it independently is rejected by B6.
    assert '--dec0-mode' not in flags
    assert m10.MODES['fp16'] == []


class _Done:
    returncode = 0
    stdout = ''
    stderr = ''


def _fake_run(calls, status='NOT-EXECUTED'):
    def fake(command, **kwargs):
        calls.append(command)
        out = Path(command[command.index('--out') + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / 'b6.json').write_text(json.dumps(
            {'execution': {'status': status, 'reasons': []},
             'fidelity': {'regression': {'status': 'PASS'}}}))
        return _Done()
    return fake


def test_run_frame_wires_colour_and_mode(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(m10.subprocess, 'run', _fake_run(calls))
    m10.run_frame('int8', tmp_path / 'c.bin', tmp_path / 'i8', tmp_path / 'L')
    m10.run_frame('fp16', tmp_path / 'c.bin', tmp_path / 'f16', tmp_path / 'L')
    int8, fp16 = calls
    for command in (int8, fp16):
        assert command[command.index('--colour') + 1] == str(tmp_path / 'c.bin')
        assert '--no-repeat' in command
    assert '--cumulative-layers' in int8 and '--cumulative-layers' not in fp16
    assert '--colour' in int8 and '--colour' in fp16


def test_run_frame_rejects_an_execution_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(m10.subprocess, 'run', _fake_run([], status='FAIL'))
    with pytest.raises(RuntimeError):
        m10.run_frame('int8', tmp_path / 'c.bin', tmp_path / 'o', tmp_path / 'L')


def test_inventory_documents_every_captured_temporal_edge():
    text = (Path(__file__).resolve().parents[1] / 'docs' / 'experiments' /
            'EXP-030-temporal-state-inventory.md').read_text(encoding='utf-8')
    for role in ('post` `+96`', 'post` `+112`', 'dec0` `+128`',
                 'enc0` `+280/288/296`', 'reset = 0'):
        assert role in text, role


def test_describe_counts_non_finite_without_hiding_it():
    a = np.array([1.0, np.inf, 2.0], np.float16)
    b = np.array([1.0, np.inf, np.nan], np.float16)
    row = m10.describe(a, b)
    assert row['non_finite'] == 2
    assert row['equal_elements'] == 2 and row['differing_elements'] == 1
    assert row['max_abs'] == 0.0


def test_zero_tensors_are_finite_and_have_no_relative_error():
    a = np.zeros((4, 4), np.float16)
    row = m10.describe(a, a.copy())
    assert row['identical'] is True and row['non_finite'] == 0
    assert row['rms'] == 0.0 and row['relative_rms'] is None


def test_state_paths_follow_the_frame_layout(tmp_path):
    assert m10.state_path(tmp_path, 'int8', 3, 'post.history') == (
        tmp_path / 'int8' / 'frame-03' / 'post-native' / 'results' / 'out-history.bin')
    assert m10.state_path(tmp_path, 'fp16', 11, 'dec0.feature') == (
        tmp_path / 'fp16' / 'frame-11' / 'dec0-native' / 'results' / 'out-feat.bin')


def test_replay_verdict_detects_a_single_changed_boundary():
    original = {f'b{i}': f'h{i}' for i in range(33)}
    assert m10.boundary_mismatches(original, dict(original)) == []
    changed = dict(original)
    changed['b7'] = 'other'
    assert m10.boundary_mismatches(original, changed) == ['b7']
    missing = dict(original)
    del missing['b31']
    assert m10.boundary_mismatches(original, missing) == ['b31']


def test_state_replay_requires_every_tensor_hash():
    original = {role: {'sha256': f'h-{role}'} for role in m10.STATE}
    replay = {role: {'sha256': f'h-{role}'} for role in m10.STATE}
    assert m10.state_matches(original, replay) is True
    replay['dec0.feature']['sha256'] = 'different'
    assert m10.state_matches(original, replay) is False


def test_the_sequence_is_tied_to_the_accepted_m09_hashes():
    """Frame 0 must stay the accepted M09 frame, not drift with the fixtures."""
    accepted = {
        'fp16': 'dc97c4cbe99346111870507735b9c7bc70846f7c7bf163f1c531ddb810d6e95b',
        'int8': 'f76ac032f45a91dc6c4ead22979eff16ed1d04c4a4cf1add2ff1adc0bd4d6c00',
    }
    text = (Path(__file__).resolve().parents[1] / 'scripts' /
            'm10_temporal.py').read_text(encoding='utf-8')
    for mode, digest in accepted.items():
        assert digest in text, mode
