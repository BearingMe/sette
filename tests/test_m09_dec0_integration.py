"""EXP-025 mode, fixture and handoff contracts without proprietary data."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import m_b6_frame as b6
import m_stage_fixture as sf


def fixture(tmp_path, reset=1, size=None):
    (tmp_path/'in-weights.bin').write_bytes(bytes(8448))
    # Deliberately no tokens file: preparation must never read activations.
    (tmp_path/'fixture.json').write_text(json.dumps({'stage':'dec0','params':{'size':size or [960,544],'reset':reset}}))
    text = ('version 2\nshader dec0_cs.cso\nthreads 64 1 1\ngroups 2040 1 1\nparams params.bin 32\n'
            'srv buffer 0 in-weights.bin 8448\nsrv buffer 1 in-tokens.bin 8355840\n'
            'uav tex2d 2 R32_UINT 960 540 out-gate.bin\n'
            'uav tex2d 3 R32_UINT 960 540 out-mix.bin\n'
            'uav tex2d 4 R32_UINT 960 540 out-feat.bin\n'
            'uav tex2d 5 R16G16B16A16_FLOAT 960 540 out-cov.bin\n')
    (tmp_path/'stage.txt').write_text(text)
    return text


def test_fp16_preparation_is_a_byte_preserving_noop(tmp_path):
    text = fixture(tmp_path)
    before = {p.name:p.read_bytes() for p in tmp_path.iterdir()}
    sf.prepare_dec0_mode(tmp_path)
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}


def test_experimental_mode_preserves_surface_descriptors_and_does_not_read_tokens(tmp_path):
    text=fixture(tmp_path)
    sf.prepare_dec0_mode(tmp_path,'int8-expansion')
    after=(tmp_path/'stage.txt').read_text()
    assert [l for l in text.splitlines() if l.startswith('uav ')]==[l for l in after.splitlines() if l.startswith('uav ')]
    assert 'shader dec0_q8_epilogue.cso' in after
    assert 'dec0_int8 int8-wc.bin int8-ws.bin int8-bias.bin' in after
    assert (tmp_path/'int8-wc.bin').stat().st_size==5120
    assert np.array_equal(np.fromfile(tmp_path/'int8-ws.bin',np.float32),np.ones(80,np.float32))
    assert (tmp_path/'int8-bias.bin').stat().st_size==320


@pytest.mark.parametrize('reset,size',[(0,[960,544]),(1,[960,540]),(1,[1920,1080])])
def test_int8_rejects_unsupported_geometry_and_temporal_frames(tmp_path,reset,size):
    fixture(tmp_path,reset,size)
    with pytest.raises(ValueError): sf.prepare_dec0_mode(tmp_path,'int8-expansion')


def test_modes_cannot_be_combined():
    b6.validate_precision_modes('fp16','fp16')
    b6.validate_precision_modes('int8-q','fp16')
    b6.validate_precision_modes('fp16','int8-expansion')
    with pytest.raises(SystemExit): b6.validate_precision_modes('int8-q','int8-expansion')


def test_internal_handoff_buffers_are_declared_for_boundary_checks(tmp_path):
    fixture(tmp_path)
    original=b6.declared_outputs(tmp_path/'stage.txt')
    sf.prepare_dec0_mode(tmp_path,'int8-expansion')
    assert b6.declared_outputs(tmp_path/'stage.txt')=={**original,**sf.DEC0_DIAGNOSTICS}


def test_precision_unknown_mode_rejected(tmp_path):
    with pytest.raises(ValueError): sf.prepare_dec0_mode(tmp_path,'int8-all')


def test_repeated_mode_preparation_is_rejected(tmp_path):
    fixture(tmp_path)
    sf.prepare_dec0_mode(tmp_path,'int8-expansion')
    with pytest.raises(ValueError): sf.prepare_dec0_mode(tmp_path,'int8-expansion')


def test_nonfinite_weights_are_rejected_before_export(tmp_path):
    fixture(tmp_path)
    data=bytearray(8448); data[0]=127
    (tmp_path/'in-weights.bin').write_bytes(data)
    with pytest.raises(ValueError): sf.prepare_dec0_mode(tmp_path,'int8-expansion')
