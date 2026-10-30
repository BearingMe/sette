"""M12 profiling contract: aggregation, graph parsing and accepted hashes.

The GPU timings themselves come from executed runs; these tests fix the pure
parts that turn timestamp files into a comparison, so a silent change to the
aggregation or the traffic join fails here.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import m12_analyze as analyze
import m12_profile as profile


def test_stats_median_and_percentile_bounds():
    row = profile.stats([5.0, 1.0, 3.0, 2.0, 4.0])
    assert row['median'] == 3.0
    assert row['min'] == 1.0 and row['max'] == 5.0
    assert row['samples'] == 5
    assert 1.0 <= row['p95'] <= 5.0


def test_family_splits_on_the_first_dot_only():
    assert analyze.family('enc1.mlp-up') == 'mlp-up'
    assert analyze.family('tube#6.attention-pv') == 'attention-pv'
    assert analyze.family('main') == 'main'


def test_parse_graph_reads_resources_stages_and_bindings(tmp_path):
    graph = tmp_path / 'graph.txt'
    graph.write_text(
        'version 1\n'
        'resource a 1024\n'
        'resource b 2048\n'
        'stage enc1.q\n'
        '  shader q.cso\n'
        '  groups 8 1 1\n'
        '  srv 0 a\n'
        '  srv 1 b\n'
        '  uav 0 b\n'
        'stage enc1.v\n'
        '  groups 4 2 1\n'
        '  srv 0 a\n'
        '  uav 0 a\n')
    resources, stages = analyze.parse_graph(graph)
    assert resources == {'a': 1024, 'b': 2048}
    assert [s['name'] for s in stages] == ['enc1.q', 'enc1.v']
    assert stages[0]['srv'] == ['a', 'b'] and stages[0]['uav'] == ['b']
    assert stages[1]['groups'] == [4, 2, 1]


def test_accepted_frame_hashes_are_the_m09_hashes():
    assert profile.ACCEPTED == {
        'fp16': 'dc97c4cbe99346111870507735b9c7bc70846f7c7bf163f1c531ddb810d6e95b',
        'int8': 'f76ac032f45a91dc6c4ead22979eff16ed1d04c4a4cf1add2ff1adc0bd4d6c00',
    }


def test_timed_and_plain_modes_keep_the_same_mode_flags():
    assert profile.MODES['fp16'] == []
    flags = profile.MODES['int8']
    assert flags[flags.index('--cumulative-mode') + 1] == 'spatial'
    assert flags[flags.index('--attention-mode') + 1] == 'int8'
