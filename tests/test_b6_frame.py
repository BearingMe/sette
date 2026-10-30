"""B6 runner contract: execution gating, metric definitions and reuse safety.

The tests are built from the declared contract rather than from a captured frame,
so a change that lets an incomplete or stale run pass fails here.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('PIL')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import m_b6_frame as b6

SHAPE = (64, 96, 4)


def frame(value=0.5, alpha=1.0):
    data = np.full(SHAPE, value, np.float16)
    data[..., 3] = alpha
    return data


def report(**over):
    """A minimal report that satisfies every declared execution check."""
    row = {'stages': {name: f'/tmp/{name}' for name in b6.EXECUTION['stages']},
           'missing_outputs': [],
           'frame': {'height': b6.H, 'width': b6.W, 'channels': 4, 'dtype': 'float16'},
           'non_finite': {'dec1': 0, 'frame': 0, 'saturated': 0},
           'repeat': {'fresh': True, 'deterministic': True, 'mismatched': []},
           'boundaries': {f'boundary-{i}': 'x' for i in range(8)}}
    row.update(over)
    return row


# ---- metric definitions ------------------------------------------------------

def test_identical_frames_are_exact():
    a = frame()
    row = b6.metrics(a, a)
    assert row['mean_abs'] == 0.0 and row['max_abs'] == 0.0
    assert row['psnr_peak1_db'] is None and row['psnr_peak_reference_db'] is None
    assert row['ssim_windowed_luma'] == pytest.approx(1.0)
    assert row['alpha_identical'] == row['alpha_samples']
    assert row['nan_or_inf_pairs'] == 0
    assert row['data_range'] == 1.0


def test_psnr_follows_the_constant_error():
    a, b = frame(0.5), frame(0.5 + 2 ** -8)
    row = b6.metrics(a, b)
    assert row['psnr_peak1_db'] == pytest.approx(-20 * np.log10(2 ** -8), abs=1e-6)
    assert row['mean_abs'] == pytest.approx(2 ** -8, rel=1e-3)
    assert row['signed_mean'] > 0


def test_non_finite_pairs_are_excluded_and_counted_not_zeroed():
    a = frame(0.5)
    b = frame(0.5 + 2 ** -8)
    b[0, 0, 0] = np.nan
    b[1, 1, 0] = np.inf
    row = b6.metrics(a, b)
    assert row['nan_or_inf_pairs'] == 2
    assert row['nan_or_inf_test'] == 2
    assert row['excluded_non_finite_pairs'] == 2
    # the remaining pairs keep their own error; a zero fill would have pulled
    # mean_abs down toward 2**-8 * (1 - 2/n)
    assert row['mean_abs'] == pytest.approx(2 ** -8, rel=1e-6)


def test_saturation_is_counted_separately_from_non_finite():
    a, b = frame(0.5), frame(0.5 + 2 ** -8)
    b[2, 2, 0] = 65504.0
    row = b6.metrics(a, b)
    assert row['saturated_test'] == 1
    assert row['saturated_reference'] == 0
    assert row['nan_or_inf_pairs'] == 0
    # the saturated sample is a real, finite, very large error
    assert row['max_abs'] == pytest.approx(65503.5)


def test_alpha_is_reported_separately_from_rgb():
    a, b = frame(0.5, 1.0), frame(0.5, 0.5)
    row = b6.metrics(a, b)
    assert row['mean_abs'] == 0.0 and row['max_abs'] == 0.0
    assert row['alpha_mean_abs'] == pytest.approx(0.5)
    assert row['alpha_identical'] == 0


def test_spatial_distribution_locates_the_error_block():
    a = frame()
    b = frame()
    b[32:48, 16:32, :3] += 0.25
    row = b6.metrics(a, b)
    assert row['block_mean_abs_max'] == pytest.approx(0.25, rel=1e-3)
    assert row['block_mean_abs_min'] == 0.0
    assert row['worst_blocks_yx'][0] == [2, 1]


def test_border_and_interior_are_measured_separately():
    a = frame()
    b = frame()
    b[0:4, :, :3] += 0.5
    row = b6.metrics(a, b, border=8)
    assert row['border_mean_abs'] > row['interior_mean_abs']


# ---- SSIM implementation -----------------------------------------------------

def test_ssim_is_one_only_for_identical_luma():
    a = frame()
    assert b6.ssim_windowed_luma(a, a) == pytest.approx(1.0)


def test_windowed_ssim_separates_brightness_from_structure():
    a = frame()
    flat = frame(0.6)
    structured = frame()
    structured[::2, :, :3] = 0.9
    assert b6.ssim_windowed_luma(a, flat) < 1.0
    assert b6.ssim_windowed_luma(a, structured) < b6.ssim_windowed_luma(a, flat)


def test_the_two_ssim_implementations_are_distinct_and_labelled():
    a = frame()
    b = frame()
    b[::4, :, :3] = 0.8
    row = b6.metrics(a, b)
    assert 'windowed' in row['ssim_algorithm']
    assert '11x11' in row['ssim_algorithm']
    # the legacy block metric averages per-block means and is not the same number
    assert row['ssim_windowed_luma'] != row['ssim_block8_luma_legacy']


# ---- execution gating --------------------------------------------------------

def test_a_complete_report_passes_execution():
    assert b6.judge_execution(report())['status'] == 'PASS'


@pytest.mark.parametrize('field,value', [
    ('missing_outputs', ['native/dec0/out-gate']),
    ('frame', {'height': 540, 'width': 960, 'channels': 4, 'dtype': 'float16'}),
    ('boundaries', {'only': 'one'}),
])
def test_incomplete_execution_fails(field, value):
    row = b6.judge_execution(report(**{field: value}))
    assert row['status'] == 'FAIL' and row['reasons']


def test_a_missing_stage_fails():
    stages = {name: '/tmp/x' for name in b6.EXECUTION['stages'] if name != 'dec0'}
    row = b6.judge_execution(report(stages=stages))
    assert row['status'] == 'FAIL'


@pytest.mark.parametrize('counts', [{'dec1': 1}, {'frame': 1}, {'saturated': 1}])
def test_non_finite_or_saturation_fails(counts):
    bad = {'dec1': 0, 'frame': 0, 'saturated': 0}
    bad.update(counts)
    row = b6.judge_execution(report(non_finite=bad))
    assert row['status'] == 'FAIL'


def test_nondeterminism_fails():
    row = b6.judge_execution(report(repeat={'fresh': True, 'deterministic': False,
                                            'mismatched': ['dec0/out-gate']}))
    assert row['status'] == 'FAIL' and 'dec0/out-gate' in row['reasons'][0]


def test_a_reused_run_cannot_reach_pass():
    row = b6.judge_execution(report(repeat={'fresh': False, 'deterministic': True,
                                            'inherited': True}))
    assert row['status'] == 'NOT-EXECUTED'
    assert row['status'] != 'PASS'


# ---- reuse safety ------------------------------------------------------------

def test_reuse_accepts_identical_inputs_and_hashes():
    previous = {'inputs': {'probe_sha256': 'a'}, 'repeat': {'hashes': {'x': '1'}}}
    assert b6.verify_reuse(previous, {'probe_sha256': 'a'}, {'x': '1'}) == {'x': '1'}


def test_reuse_refuses_changed_inputs():
    previous = {'inputs': {'probe_sha256': 'a'}, 'repeat': {'hashes': {'x': '1'}}}
    with pytest.raises(SystemExit):
        b6.verify_reuse(previous, {'probe_sha256': 'b'}, {'x': '1'})


def test_reuse_refuses_changed_outputs():
    previous = {'inputs': {'probe_sha256': 'a'}, 'repeat': {'hashes': {'x': '1'}}}
    with pytest.raises(SystemExit):
        b6.verify_reuse(previous, {'probe_sha256': 'a'}, {'x': '2'})


def test_reuse_refuses_a_report_without_recorded_hashes():
    with pytest.raises(SystemExit):
        b6.verify_reuse({'inputs': {'probe_sha256': 'a'}}, {'probe_sha256': 'a'},
                        {'x': '1'})


# ---- declared outputs, frame geometry, exports -------------------------------

def test_declared_outputs_reads_buffers_and_texels(tmp_path):
    manifest = tmp_path / 'stage.txt'
    manifest.write_text('version 2\n'
                        'uav buffer 5 out-packed.bin 1024\n'
                        'uav tex2d 7 R32G32B32A32_FLOAT 4 2 out-half.bin\n')
    assert b6.declared_outputs(manifest) == {'packed': 1024, 'half': 4 * 2 * 16}


def test_an_unknown_uav_format_is_an_error_not_a_guess(tmp_path):
    manifest = tmp_path / 'stage.txt'
    manifest.write_text('uav tex2d 1 R11G11B10_FLOAT 2 2 out.bin\n')
    with pytest.raises(KeyError):
        b6.declared_outputs(manifest)


def test_frame_rejects_a_short_or_long_buffer(tmp_path):
    path = tmp_path / 'frame.bin'
    np.zeros(16, np.float16).tofile(path)
    with pytest.raises(RuntimeError):
        b6.frame(path)


def test_export_writes_rgba_and_records_the_difference_scale(tmp_path):
    a = frame()
    b = frame()
    b[::3, :, 0] += 0.05
    out = b6.export(tmp_path / 'frame-native', a, b)
    from PIL import Image
    for name in ('reference', 'reconstructed', 'difference'):
        image = Image.open(out['images'][name])
        assert image.mode == 'RGBA'
        assert image.size == (SHAPE[1], SHAPE[0])
    assert out['rgb_difference_scale'] == pytest.approx(0.05, rel=1e-2)


def test_regression_bounds_are_declared_and_reported():
    row = {'native': {'psnr_peak1_db': 47.0, 'ssim_windowed_luma': 0.998,
                      'mean_abs': 0.001, 'max_abs': 0.2, 'nan_or_inf_test': 0,
                      'saturated_test': 0},
           'image': {'psnr_peak1_db': 46.7, 'ssim_windowed_luma': 0.9979,
                     'mean_abs': 0.0012, 'max_abs': 0.15, 'nan_or_inf_test': 0,
                     'saturated_test': 0}}
    gap = {'image-vs-native': {'psnr_peak1_db': 46.5}}
    verdict = b6.judge_regression({'comparisons': row, 'chain_gap': gap})
    assert verdict['status'] == 'PASS'
    assert verdict['bounds'] == b6.REGRESSION


def test_a_worse_than_control_native_frame_fails_regression():
    row = {'native': {'psnr_peak1_db': 30.0, 'ssim_windowed_luma': 0.9,
                      'mean_abs': 0.05, 'max_abs': 2.0, 'nan_or_inf_test': 0,
                      'saturated_test': 0}}
    verdict = b6.judge_regression({'comparisons': row, 'chain_gap': {}})
    assert verdict['status'] == 'FAIL'
    assert any('native.psnr_peak1_db' in r for r in verdict['reasons'])

