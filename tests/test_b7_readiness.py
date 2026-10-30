"""B7 readiness audit: manifest parsing and handoff detection.

Synthetic capture folders, so the audit's claims are testable without the
proprietary image dumps.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import m_b7_readiness as b7

MANIFEST = """kernel rrlite_post_3_1_mvhi_ldr_folded
launch 180 51 1 16 16 1 0
args 248
image 0 tex 0x1 0x2 0x0 2880 1620 1 0x10 4 0 23040 37324800 0
image-bind 0 56 tex
image 1 surf 0x3 0x4 0x0 2880 1620 1 0x10 4 0 23040 37324800 0
image-bind 1 96 surf
"""


def build(root, frames, history_in, history_out):
    """One capture folder per frame, with the given bytes in the two offsets."""
    for frame in frames:
        directory = root / f'replay-{frame:06d}-rrlite_post_3_1_mvhi_ldr_folded'
        directory.mkdir(parents=True)
        (directory / 'manifest.txt').write_text(MANIFEST)
        (directory / 'image-0.bin').write_bytes(history_in[frame])
        (directory / 'image-1.bin').write_bytes(history_out[frame])
    return root


def test_manifest_roles_resolve_by_parameter_offset(tmp_path):
    directory = tmp_path / 'replay-000001-rrlite_post_3_1_mvhi_ldr_folded'
    directory.mkdir()
    (directory / 'manifest.txt').write_text(MANIFEST)
    kernel, images = b7.parse_manifest(directory / 'manifest.txt')
    assert kernel == 'post_3_1_mvhi_ldr_folded'
    assert images[56]['kind'] == 'tex' and images[96]['kind'] == 'surf'
    assert images[56]['width'] == 2880 and images[56]['height'] == 1620
    assert images[56]['bytes'] == 37324800
    assert images[56]['format'] == '0x10'


def test_the_handoff_is_found_when_the_bytes_match(tmp_path):
    payload = np.arange(16, dtype=np.uint8).tobytes()
    other = np.flip(np.arange(16, dtype=np.uint8)).tobytes()
    root = build(tmp_path, [1, 2, 3],
                 history_in={1: other, 2: payload, 3: other},
                 history_out={1: other, 2: other, 3: payload})
    report = b7.audit(root)
    handoff = report['handoffs'][0]
    assert handoff['observed'] and handoff['size_match']
    pairs = {(p['producer_frame'], p['consumer_frame']) for p in handoff['equal_pairs']}
    assert (3, 2) in pairs


def test_a_zero_frame_is_reported_as_zero_not_as_evidence(tmp_path):
    zero = bytes(16)
    payload = np.arange(16, dtype=np.uint8).tobytes()
    root = build(tmp_path, [1, 2], history_in={1: zero, 2: payload},
                 history_out={1: zero, 2: payload})
    report = b7.audit(root)
    pairs = report['handoffs'][0]['equal_pairs']
    assert any(p['producer_frame'] == 1 and p['consumer_frame'] == 1 and p['all_zero']
               for p in pairs)
    assert not any(p['producer_frame'] == 2 and p['all_zero'] for p in pairs)
    roles = report['kernels']['post_3_1_mvhi_ldr_folded']['roles']
    assert roles['history_out']['per_frame'][1]['all_zero'] is True
    assert roles['history_in']['per_frame'][2]['all_zero'] is False


def test_geometry_and_bytes_are_reported_per_role(tmp_path):
    payload = b'\x01' * 16
    root = build(tmp_path, [1], history_in={1: payload}, history_out={1: payload})
    roles = b7.audit(root)['kernels']['post_3_1_mvhi_ldr_folded']['roles']
    assert roles['history_in']['geometry'] == ['2880x1620']
    assert roles['history_in']['bytes'] == [37324800]
    assert roles['history_in']['distinct_contents'] == 1


def test_a_missing_role_is_reported_and_does_not_crash(tmp_path):
    """A stage whose manifest lacks a temporal binding is reported, not guessed."""
    directory = tmp_path / 'replay-000001-rrlite_dec0_4x4_folded'
    directory.mkdir()
    (directory / 'manifest.txt').write_text(
        'kernel rrlite_dec0_4x4_folded\n'
        'image 0 surf 0x1 0x2 0x0 960 540 1 0x08 4 200 3840 2073600 0\n'
        'image-bind 0 128 surf\n')
    (directory / 'image-0.bin').write_bytes(bytes(2073600))
    report = b7.audit(tmp_path)
    missing = {row['role'] for row in report['missing_roles']}
    assert missing == {'feature_in_f16', 'feature_in_snorm8'}
    handoff = [h for h in report['handoffs'] if h['consumer'].startswith('dec0')][0]
    assert handoff['observed'] is False and handoff['missing'] == ['consumer']


def test_an_empty_capture_is_an_empty_report(tmp_path):
    report = b7.audit(tmp_path)
    assert report['kernels'] == {}
    assert all(not row['observed'] for row in report['handoffs'])
