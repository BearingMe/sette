"""M15-B: fused MLP GELU/quantizer graph contract and operator oracle."""
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import m09_mlp_up as up
import m15_arena as arena
import m_layer_suite as suite

TEMP = Path.home() / 'AppData/Local/Temp/opencode'
LAYERS = TEMP / 'm09h-complete-chain-20261030/layers'
PROBE = Path(__file__).resolve().parents[1] / 'build/native/Release/m_network_stage.exe'
SHADERS = Path(__file__).resolve().parents[1] / 'build/native'


def stages():
    def stage(name, shader, groups, srvs, uavs, params='params.bin'):
        return {'name': name, 'lines': [
            ['shader', shader], ['threads', '64', '1', '1'],
            ['groups', str(groups), '1', '1'], ['params', params, '16']]
            + [['srv', str(i), v] for i, v in enumerate(srvs)]
            + [['uav', str(i), v] for i, v in enumerate(uavs)]}
    return [
        stage('enc1.mlp-up', 'swin_mlp_up_q8.cso', 34160,
              ['a', 'w', 'as', 'ws', 'bias'], ['z', 'acc', 'float']),
        stage('enc1.mlp-gelu', 'swin_mlp_gelu.cso', 2135,
              ['z', 'weights', 'x2'], ['g', 'scratch']),
        stage('enc1.mlp-down-quant', 'swin_q8_quant.cso', 2135,
              ['g'], ['codes', 'scales']),
        stage('enc1.mlp-down', 'swin_mlp_down_q8.cso', 8540,
              ['codes', 'w', 'scales', 'ws', 'res'], ['out', 'acc', 'float']),
    ]


def test_fuse_preserves_dependencies_and_removes_both_stages():
    fused, count = suite.fuse_mlp_stages(stages(), layers={'enc1'})
    assert count == 1
    assert [s['name'] for s in fused] == [
        'enc1.mlp-up', 'enc1.mlp-fused', 'enc1.mlp-down']
    body = dict((w[0], w[2]) for w in fused[1]['lines'] if w[0] in ('srv', 'uav'))
    assert body == {'srv': 'z', 'uav': 'scales'}
    lines = fused[1]['lines']
    assert ['threads', '256', '1', '1'] in lines
    assert ['groups', '2135', '1', '1'] in lines
    assert ['uav', '0', 'g'] in lines
    assert ['uav', '1', 'codes'] in lines
    assert ['uav', '2', 'scales'] in lines
    assert not any(s['name'].endswith(('.mlp-gelu', '.mlp-down-quant'))
                   for s in fused)


def test_fused_chain_matches_locked_stage_and_resource_counts(tmp_path):
    if not LAYERS.is_dir():
        pytest.skip('accepted layer fixtures are not present')
    out = tmp_path / 'chain'
    info = suite.build_chain(str(LAYERS), str(out), fuse_mlp=True)
    assert info['stages'] == 224 and info['resources'] == 680
    assert info['fused_mlp'] == 10
    resources, parsed, dumps = up.suite._parse_graph(out / 'graph.txt')
    names = [s['name'] for s in parsed]
    assert names.count('enc1.mlp-fused') == 1
    assert sum(n.endswith('.mlp-fused') for n in names) == 10
    assert not any(n.endswith(('.mlp-gelu', '.mlp-down-quant')) for n in names)
    for stage in parsed:
        if not stage['name'].endswith('.mlp-fused'):
            continue
        lines = stage['lines']
        assert ['threads', '256', '1', '1'] in lines
        assert [w[1] for w in lines if w[0] == 'shader'] == [
            'swin_mlp_gelu_quant.cso']
        assert len([w for w in lines if w[0] == 'uav']) == 3
    assert len(dumps) == 13


def test_fused_graph_keeps_the_arena_plan(tmp_path):
    if not LAYERS.is_dir():
        pytest.skip('accepted layer fixtures are not present')
    control = tmp_path / 'control'
    fused = tmp_path / 'fused'
    suite.build_chain(str(LAYERS), str(control))
    suite.build_chain(str(LAYERS), str(fused), fuse_mlp=True)
    a = arena.analyse(control / 'graph.txt')['arena']
    b = arena.analyse(fused / 'graph.txt')['arena']
    for key in ('heap_bytes', 'persistent_bytes', 'placed_resources',
                'peak_live_bytes', 'peak_transient_bytes',
                'largest_transient_bytes'):
        assert a[key] == b[key], key
    assert b['placed_resources'] == 516


def test_operator_oracle_control_and_fused_are_byte_identical(tmp_path):
    if not (PROBE.is_file() and (SHADERS / 'swin_mlp_gelu_quant.cso').is_file()):
        pytest.skip('D3D12 probe or fused shader is not built')
    out = tmp_path / 'oracle'
    done = __import__('subprocess').run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts/m15b_fusion.py'),
         '--out', str(out), '--probe', str(PROBE), '--shaders', str(SHADERS)],
        capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-1200:] + done.stderr[-1200:]
    report = json.loads((out / 'oracle.json').read_text())
    assert report['status'] == 'PASS'
    assert report['checks']['g_identical']
    assert report['checks']['codes_identical']
    assert report['checks']['scales_identical']
