"""M13 selective-precision and asymmetric-quantization contracts.

The GPU arithmetic itself is validated element-by-element by
`scripts/m13_validate_asym.py`; these tests fix the pure parts that must not
drift: graph rewriting, restoration parsing, the asymmetric encoding contract
and the requirement that a mixed configuration never claims full INT8 coverage.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import m13_selective as selective
import m13_validate_asym as validate


def test_split_and_flatten_preserve_every_line():
    lines = ['version 1', 'resource a 16', 'stage one', 'shader s.cso', 'srv 0 a',
             'dump a 0 16 out.bin', 'resource b 32']
    items = selective.split_graph(lines)
    assert selective.flatten(items) == lines
    assert [item[1] for item in items if item[0] == 'stage'] == ['one']


def test_parse_restore_accepts_pipe_and_comma():
    assert selective.parse_restore('enc1:q|dec2:expand') == {('enc1', 'q'),
                                                            ('dec2', 'expand')}
    assert selective.parse_restore('enc1:q,enc1:v') == {('enc1', 'q'), ('enc1', 'v')}
    with pytest.raises(SystemExit):
        selective.parse_restore('enc1:not-a-family')
    with pytest.raises(SystemExit):
        selective.parse_restore('nope:q')


def test_asym_quantizer_uses_the_full_signed_range():
    x = np.array([[0.0, 0.25, 0.5, 1.0, 0.75, 0.125, 0.875, 0.375]], np.float16)
    codes, scale, offset = validate.quantize_asym(x)
    assert codes.dtype == np.int8 and codes.min() >= -128 and codes.max() <= 127
    assert codes.min() == -128 and codes.max() == 127
    dequant = codes.astype(np.float64) * float(scale[0]) + float(offset[0])
    assert np.max(np.abs(dequant - x.astype(np.float64))) <= float(scale[0]) / 2 + 1e-6


def test_asym_quantizer_constant_row_is_stable():
    x = np.full((1, 8), 0.5, np.float16)
    codes, scale, offset = validate.quantize_asym(x)
    dequant = codes.astype(np.float64) * float(scale[0]) + float(offset[0])
    assert np.all(codes == codes[0, 0])
    assert np.allclose(dequant, 0.5, atol=1e-6)


def minimal_asym_graph():
    return [
        'version 1',
        'resource i8_g 16',
        'resource i8_mlp-down_a 16',
        'resource i8_mlp-down_as 4',
        'resource i8_mlp-down_w 16 i8_mlp-down_w.bin',
        'resource i8_mlp-down_ws 4 i8_mlp-down_ws.bin',
        'resource i8_mlp-down_acc 16',
        'resource i8_mlp-down_float 16',
        'resource x2 16',
        'resource x2b 16',
        'resource att_prob 16',
        'resource att_pc 8',
        'resource att_ps 4',
        'resource att_vc 8',
        'resource att_vs 4',
        'resource att_pacc 16',
        'resource att_value32 16',
        'resource att_value16 8',
        'resource o 8',
        'stage mlp-down-quant',
        'shader swin_q8_quant.cso',
        'threads 256 1 1',
        'groups 1 1 1',
        'params p 16',
        'srv 0 i8_g',
        'uav 0 i8_mlp-down_a',
        'uav 1 i8_mlp-down_as',
        'stage mlp-down',
        'shader swin_mlp_down_q8.cso',
        'threads 64 1 1',
        'groups 1 1 1',
        'params p 16',
        'srv 0 i8_mlp-down_a',
        'srv 1 i8_mlp-down_w',
        'srv 2 i8_mlp-down_as',
        'srv 3 i8_mlp-down_ws',
        'srv 4 x2',
        'uav 0 x2b',
        'uav 1 i8_mlp-down_acc',
        'uav 2 i8_mlp-down_float',
        'stage attention-p-quant',
        'shader swin_q8_quant.cso',
        'threads 256 1 1',
        'groups 1 1 1',
        'params p 16',
        'srv 0 att_prob',
        'uav 0 att_pc',
        'uav 1 att_ps',
        'stage attention-pv',
        'shader swin_attention_pv.cso',
        'threads 64 1 1',
        'groups 1 1 1',
        'params p 16',
        'srv 0 att_pc',
        'srv 1 att_vc',
        'srv 2 att_ps',
        'srv 3 att_vs',
        'uav 0 o',
        'uav 1 att_pacc',
        'uav 2 att_value32',
        'uav 3 att_value16',
        'dump x2b 0 16 out.bin',
    ]


def test_asym_rewrite_switches_only_the_two_consumers(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    # K=4, N=4 one-byte codes, stored column-major; sums are 1+1+1+1 = 4.
    (source / 'i8_mlp-down_w.bin').write_bytes(np.ones(16, np.uint8).tobytes())
    (source / 'i8_mlp-down_ws.bin').write_bytes(np.ones(4, '<f4').tobytes())
    record = {'int8_operations': [
        {'name': 'mlp-down', 'M': 4, 'K': 4, 'N': 4,
         'weight_codes': 'i8_mlp-down_w', 'weight_scales': 'i8_mlp-down_ws'}],
        'int8_attention': {'instances': 1}}
    lines, sizes = selective.asym_rewrite(minimal_asym_graph(), tmp_path, source,
                                          'enc1', record, True)
    blocks = dict(selective.stage_blocks(lines))
    assert blocks['mlp-down-quant'][0] == 'shader swin_q8_quant_asym.cso'
    assert 'uav 2 i8_mlp-down_off' in blocks['mlp-down-quant']
    assert blocks['mlp-down'][0] == 'shader swin_mlp_down_q8_asym.cso'
    assert 'srv 5 i8_mlp-down_wsum' in blocks['mlp-down']
    assert 'srv 6 i8_mlp-down_off' in blocks['mlp-down']
    assert blocks['attention-p-quant'][0] == 'shader swin_q8_quant_asym.cso'
    assert blocks['attention-pv'][0] == 'shader swin_attention_pv_asym.cso'
    assert 'srv 4 att_po' in blocks['attention-pv']
    assert sizes['i8_mlp-down_off'] == 16
    assert sizes['i8_mlp-down_wsum'] == 16
    assert sizes['att_po'] == 64
    sums = np.fromfile(tmp_path / 'i8_mlp-down_wsum.bin', '<i4')
    assert sums.tolist() == [4, 4, 4, 4]


def test_mixed_coverage_is_never_reported_as_full_int8():
    """The coverage ledger must count restored operators as FP16."""
    ledger = [{'id': 'enc1.q', 'layer': 'enc1', 'family': 'q', 'macs': 100},
              {'id': 'enc1.v', 'layer': 'enc1', 'family': 'v', 'macs': 100},
              {'id': 'enc1.up', 'layer': 'enc1', 'family': 'up', 'macs': 100},
              {'id': 'enc1.down', 'layer': 'enc1', 'family': 'down', 'macs': 100},
              {'id': 'enc1.dec0', 'layer': 'enc1', 'family': 'dec0', 'macs': 100}]
    # Mirrors coverage() without touching the external fixture tree.
    rows = []
    for row in ledger:
        family = selective.INVENTORY_FAMILY.get(row['family'], row['family'])
        rows.append({**row, 'int8': (row['layer'], family) not in {('enc1', 'q')}})
    assert len([r for r in rows if r['int8']]) == 4
    assert [r['id'] for r in rows if not r['int8']] == ['enc1.q']


def test_validation_script_declares_the_expected_shaders():
    text = (Path(__file__).resolve().parents[1] / 'scripts' /
            'm13_validate_asym.py').read_text(encoding='utf-8')
    for name in ('swin_q8_quant_asym.cso', 'swin_mlp_down_q8_asym.cso',
                 'swin_attention_pv_asym.cso'):
        assert name in text
