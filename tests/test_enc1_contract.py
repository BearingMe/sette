"""Fail closed on unproven execution mode; no proprietary fixtures required."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from m_capture_provenance import dispatches, verify_translated, audit
from m_enc1_contract import verify_macros, verify_replay, metrics
from m_enc1_native_trace import once

K = 'rrlite_enc1_4x4'
LINE = f'[d4r-launch] kernel="{K}" backend=translated grid=61,35,1 block=32,1,2'


def test_translated_requires_actual_dispatch():
    with pytest.raises(ValueError):
        verify_translated(f'[d4r-native] miss kernel="{K}" fallback=translated', K)
    assert verify_translated(LINE, K)['backend'] == 'translated'


@pytest.mark.parametrize('extra', [LINE, LINE.replace('backend=translated', 'backend=native'),
                                 f'[d4r-native] module_direct kernel="{K}" translated_jit=skipped'])
def test_translated_rejects_multiple_or_native(extra):
    with pytest.raises(ValueError):
        verify_translated(LINE + '\n' + extra, K)


def test_source_defines_do_not_prove_effective_mode():
    fast = '#define SWIN_NO_Q8\n#define SWIN_F32ACC\n'
    exact = '#define D4R_ACCURACY\n#define SWIN_EXACT\n'
    verify_macros(fast, 'fast')
    verify_macros(exact, 'conservative')
    for text, mode in [(fast, 'conservative'), (exact, 'fast'), (fast + exact, 'conservative')]:
        with pytest.raises(ValueError):
            verify_macros(text, mode)


def test_replay_requires_successful_load_and_dispatch():
    with pytest.raises(ValueError):
        verify_replay(f'PASS NATIVE_REPLAY architecture=gfx1101 kernel={K} iterations=1')


def test_audit_cannot_infer_backend_from_manifest(tmp_path):
    capture = tmp_path / ('replay-000001-' + K)
    capture.mkdir()
    (capture / 'manifest.txt').write_text('kernel ' + K)
    with pytest.raises(ValueError):
        audit('', tmp_path)
    assert dispatches(LINE)[0]['block'] == '32,1,2'


def test_audit_binds_each_completion_and_handles_wrapped_path(tmp_path):
    capture = tmp_path / ('replay-000001-' + K)
    capture.mkdir()
    (capture / 'manifest.txt').write_text('kernel ' + K)
    begin = f'[d4r-replay] begin kernel={K} allocations=2 path={capture}\n'
    complete = f'\n[d4r-replay] complete \npath={capture}\n'
    assert audit(begin + LINE + complete, tmp_path)[0]['capture'] == capture.name
    with pytest.raises(ValueError):
        audit(begin + LINE, tmp_path)
    with pytest.raises(ValueError):
        audit(begin + LINE + complete.replace('000001', '000002'), tmp_path)


def test_metrics_use_decoded_values_not_code_distance():
    row = metrics(np.array([0, 128], np.uint8), np.array([128, 0], np.uint8))
    assert row['differing_codes'] == 2
    assert row['max_abs_decoded'] == 0
    with pytest.raises(ValueError):
        metrics(np.array([], np.uint8), np.array([], np.uint8))


def test_instrumentation_rejects_missing_or_ambiguous_anchor():
    for value in ('', 'anchor anchor'):
        with pytest.raises(ValueError):
            once(value, 'anchor', 'replacement')
