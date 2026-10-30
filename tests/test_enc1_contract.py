"""Fail closed on unproven execution mode; no proprietary fixtures required."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from m_capture_provenance import dispatches, verify_translated, audit
from m_enc1_contract import (verify_macros, verify_replay, metrics, natural,
                             distribution)
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


def test_translated_dispatch_count_is_explicit():
    two = LINE + '\n' + LINE.replace('grid=61,35,1', 'grid=31,18,1')
    with pytest.raises(ValueError):
        verify_translated(two, K)
    assert [r['grid'] for r in verify_translated(two, K, 2)] == ['61,35,1', '31,18,1']
    with pytest.raises(ValueError):
        verify_translated(two.replace('backend=translated', 'backend=native', 1), K, 2)


def test_replay_geometry_follows_the_requested_kernel():
    kernel = 'rrlite_enc2_4x4'
    base = ('hipModuleLoad(replay) -> 0 (hipSuccess)\n'
            'hipModuleLaunchKernel(replay prep) -> 0 (hipSuccess)\n'
            'hipModuleLaunchKernel(replay) -> 0 (hipSuccess)\n'
            f'PASS NATIVE_REPLAY architecture=gfx1101 kernel={kernel} iterations=1\n')
    verify_replay(base + f'REPLAY kernel={kernel} grid=31,18,1 block=32,1,4', kernel)
    with pytest.raises(ValueError):
        verify_replay(base, kernel)
    verify_replay(base.replace(kernel, K) +
                  f'REPLAY kernel={K} grid=61,35,1 block=32,1,4')


def test_distribution_locates_the_differing_channel_and_token():
    W, H, C = 16, 8, 32
    codes = np.zeros((H, W, C), np.uint8)
    storage = codes.reshape(H, W, C // 32, 32).transpose(2, 0, 1, 3).reshape(-1).copy()
    assert np.array_equal(natural(storage, W, H, C), codes)
    codes[2, 5, 3] = 0x40
    other = codes.reshape(H, W, C // 32, 32).transpose(2, 0, 1, 3).reshape(-1)
    row = distribution(storage, other, W, H, C)
    assert row['channel_exact_worst'] == 3
    assert row['channels_below_99'] == 0
    assert row['token_exact_worst_yx'] == [2, 5]
    assert row['channel_exact_min'] == 1.0 - 1.0 / (H * W)
    codes[:, :, 3] = 0x41
    column = codes.reshape(H, W, C // 32, 32).transpose(2, 0, 1, 3).reshape(-1)
    assert distribution(storage, column, W, H, C)['channels_below_99'] == 1


def test_metrics_report_bias_saturation_and_nonfinite_pairs():
    row = metrics(np.array([0x7e, 0x7f], np.uint8), np.array([0x7e, 0x3c], np.uint8))
    assert row['saturated_reference'] == 1 and row['saturated_actual'] == 1
    assert row['nonfinite_pairs'] == 1
    assert row['signed_mean_decoded'] == 0.0
