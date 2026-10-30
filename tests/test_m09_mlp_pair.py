"""EXP-027: non-proprietary down-projection arithmetic and graph contracts."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import m09_mlp_pair as pair
import m_b6_frame as b6


def test_down_scalar_integer_and_residual_oracle():
    rng = np.random.default_rng(27)
    g = rng.normal(size=(4, 256)).astype('<f2')
    g[0] = 0
    w = rng.normal(size=(256, 64)).astype('<f2').astype('<f4')
    w[:, 0] = 0
    residual = rng.normal(size=(4, 64)).astype('<f2')
    codes, scales, wc, ws, acc, product, y, half = pair.cpu_down(g, w, residual)
    assert np.array_equal(half[0], residual[0])
    assert np.array_equal(half[:, 0], residual[:, 0])
    for row in range(4):
        c, s = pair.up.scalar_row(g[row])
        assert np.array_equal(c, codes[row]) and s == scales[row]
        for col in (0, 1, 31, 63):
            total = sum(int(c[k])*int(wc[k, col]) for k in range(256))
            assert total == acc[row, col]
            p = np.float32(np.float32(np.float32(total)*s)*ws[col])
            value = np.float32(p + np.float32(residual[row, col]))
            assert p.tobytes() == product[row, col].tobytes()
            assert value.tobytes() == y[row, col].tobytes()
            assert np.float16(value).tobytes() == half[row, col].tobytes()


def test_down_int32_bound_and_signed_extrema():
    for sign in (-1, 1):
        assert pair.up.eq.scalar_dot(np.full(256, sign*127), np.full(256, 127)) == sign*4129024
    assert 256*127*127 < 2**24 < 2**31


def test_down_physical_mapping_nonuniform_bijection():
    base = 16000
    offsets = np.array([[pair.raw_down_offset(base, k, c) for c in range(64)] for k in range(32)])
    reference = pair.up.lf.lo.sm.woff_table(base, 0, 512, 32, 64)[:, [pair.up.gp(c) for c in range(64)]]
    assert np.array_equal(offsets, reference)
    assert np.unique(offsets).size == 2048
    source = np.arange(20000)*73+19
    assert np.array_equal(source[offsets], source[reference])


def fixture(directory):
    directory.mkdir()
    r = {'layer': 'enc1', 'geometry': {'C': 64, 'tokens': 2}, 'q_mode': 'fp16',
         'weights': {'offsets': {'w1': 0, 'b1': 16384, 'w2': 16640}}}
    (directory / 'fixture.json').write_text(json.dumps(r))
    (directory / 'weights.bin').write_bytes(np.zeros(33024, '<f4').tobytes())
    (directory / 'graph.txt').write_text(
        'version 1\nresource h2 256\nresource x2 256\nresource x2b 256\nresource weights 132096 weights.bin\n'
        'resource scratch 16\nstage mlp\nshader swin_mlp.cso\nthreads 64 1 1\ngroups 1 1 1\n'
        'params params-mlp.bin 160\nsrv 0 h2\nsrv 1 weights\nsrv 2 x2\nuav 0 x2b\nuav 1 scratch\n'
        'stage store\nshader swin_store.cso\nthreads 64 1 1\ngroups 1 1 1\nparams params-store.bin 160\n'
        'srv 0 x2b\nuav 0 h2\nuav 1 scratch\n')
    return directory


@pytest.mark.parametrize('fp16_down', [False, True])
def test_pair_fixture_preserves_gpu_residual_and_register_order(tmp_path, fp16_down):
    source = fixture(tmp_path / 'source')
    # No h2 or activation data exists on disk: preparation must be weights-only.
    out = pair.prepare(source, tmp_path / 'pair', fp16_down=fp16_down)
    resources, stages, dumps = pair.up.suite._parse_graph(out / 'graph.txt')
    names = [s['name'] for s in stages]
    assert names[:3] == ['mlp-quant', 'mlp-up', 'mlp-gelu']
    assert names[-1] == 'store'
    assert names[3:-1] == (['mlp-down'] if fp16_down else ['mlp-down-quant', 'mlp-down'])
    for s in stages:
        for kind in ('srv', 'uav'):
            regs = [int(w[1]) for w in s['lines'] if w[0] == kind]
            assert regs == list(range(len(regs)))
    down = stages[-2]
    assert ['srv', '2' if fp16_down else '4', 'x2'] in down['lines']
    assert ['uav', '0', 'x2b'] in down['lines']
    assert 'mlp' not in names


def test_pair_mode_remains_exclusive_and_fixture_matched(tmp_path):
    b6.validate_precision_modes('fp16', 'fp16', 'int8-pair')
    with pytest.raises(SystemExit):
        b6.validate_precision_modes('int8-q', 'fp16', 'int8-pair')
    with pytest.raises(SystemExit):
        b6.validate_precision_modes('fp16', 'int8-expansion', 'int8-pair')
    source = fixture(tmp_path / 'source')
    out = pair.prepare(source, tmp_path / 'pair')
    with pytest.raises(SystemExit, match='does not match'):
        pair.up.suite.build_chain(str(tmp_path), str(tmp_path / 'chain'), mlp_int8=True,
                                 enc1_mlp=str(out), mlp_mode='int8-up')


def test_pair_chain_rejects_fp16_down_fallback(tmp_path):
    source = fixture(tmp_path / 'source')
    out = pair.prepare(source, tmp_path / 'pair')
    _, stages, _ = pair.up.suite._parse_graph(out / 'graph.txt')
    for s in stages:
        s['name'] = 'enc1.' + s['name']
    pair.up.suite.validate_mlp_stages(stages, 'int8-pair')
    for s in stages:
        if s['name'] == 'enc1.mlp-down':
            for words in s['lines']:
                if words[0] == 'shader':
                    words[1] = 'swin_mlp_down_probe.cso'
    with pytest.raises(SystemExit, match='fallback'):
        pair.up.suite.validate_mlp_stages(stages, 'int8-pair')
