"""B6 exploratory frame metrics: PSNR, SSIM, error distribution, non-finite handling."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('PIL')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from m_b6_frame import metrics, ssim_luma

SHAPE = (64, 96, 4)


def frame(value=0.5):
    data = np.full(SHAPE, value, np.float16)
    data[..., 3] = 1.0
    return data


def test_identical_frames_are_exact():
    a = frame()
    row = metrics(a, a)
    assert row['mean_abs'] == 0.0 and row['max_abs'] == 0.0
    assert row['psnr_peak1_db'] is None and row['psnr_peak_reference_db'] is None
    assert row['ssim_luma'] == pytest.approx(1.0)
    assert row['nan_or_inf_pairs'] == 0


def test_psnr_follows_the_constant_error():
    a, b = frame(0.5), frame(0.5 + 2 ** -8)
    row = metrics(a, b)
    assert row['psnr_peak1_db'] == pytest.approx(-20 * np.log10(2 ** -8), abs=1e-6)
    assert row['mean_abs'] == pytest.approx(2 ** -8, rel=1e-3)
    assert row['ssim_luma'] < 1.0
    assert row['signed_mean'] > 0


def test_non_finite_and_saturation_are_counted_not_hidden():
    a = frame()
    b = frame()
    b[0, 0, 0] = np.nan
    b[1, 1, 0] = 65504.0
    row = metrics(a, b)
    assert row['nan_or_inf_pairs'] == 1
    assert row['nan_or_inf_test'] == 1
    assert row['saturated_test'] == 1
    assert row['saturated_reference'] == 0


def test_spatial_distribution_locates_the_error_block():
    a = frame()
    b = frame()
    b[32:48, 16:32, :3] += 0.25
    row = metrics(a, b)
    assert row['block_mean_abs_max'] == pytest.approx(0.25, rel=1e-3)
    assert row['block_mean_abs_min'] == 0.0
    assert row['worst_blocks_yx'][0] == [2, 1]


def test_ssim_separates_brightness_from_structure():
    a = frame()
    flat = frame(0.6)
    structured = frame()
    structured[::2, :, :3] = 0.9
    assert ssim_luma(a, a) == pytest.approx(1.0)
    assert ssim_luma(a, flat) < 1.0
    assert ssim_luma(a, structured) < ssim_luma(a, flat)
