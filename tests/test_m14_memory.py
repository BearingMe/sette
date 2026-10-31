"""M14 memory/lifetime analysis and baseline wiring contracts.

The VRAM numbers themselves come from executed timed probes; these tests fix the
pure parts that would silently misreport them: graph parsing, live-interval
accounting, diagnostic classification, and the accepted hashes/mode flags.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import m14_baseline as baseline
import m14_memory as memory


GRAPH = """version 1
resource weights 100 weights.bin
resource a 100
resource b 100
resource c 100
resource unused 100
stage s0
shader x.cso
threads 1 1 1
groups 1 1 1
params p 16
srv 0 weights
uav 0 a
stage s1
shader y.cso
threads 1 1 1
groups 1 1 1
params p 16
srv 0 a
uav 0 b
stage s2
shader z.cso
threads 1 1 1
groups 1 1 1
params p 16
srv 0 b
uav 0 c
dump c 0 100 out.bin
"""


def test_parse_graph_reads_resources_stages_and_dumps(tmp_path):
    path = tmp_path / 'graph.txt'
    path.write_text(GRAPH)
    resources, stages, dumps = memory.parse_graph(path)
    assert resources['weights']['init'] is True
    assert [s['name'] for s in stages] == ['s0', 's1', 's2']
    assert stages[0]['srv'] == ['weights'] and stages[1]['uav'] == ['b']
    assert dumps == [{'resource': 'c', 'bytes': 100, 'file': 'out.bin'}]


def test_peak_live_differs_from_declared_and_alias_is_a_bound(tmp_path):
    path = tmp_path / 'graph.txt'
    path.write_text(GRAPH)
    report = memory.analyse(path)
    assert report['declared_bytes'] == 500       # weights + a + b + c + unused
    assert report['peak_live_bytes'] == 200      # weights+a (s0), a+b (s1), b+c (s2)
    assert report['alias_saving_bytes'] == 300
    # c is write-only and dumped, so production excludes it; weights+a and a+b remain.
    assert report['peak_production_bytes'] == 200
    assert report['diagnostic_bytes'] == 100


def test_unbound_declared_resource_is_not_live(tmp_path):
    path = tmp_path / 'graph.txt'
    path.write_text(GRAPH)
    report = memory.analyse(path)
    # `unused` is allocated but never bound, so it cannot contribute to a live peak.
    assert report['peak_live_bytes'] == 200
    assert report['declared_bytes'] == 500


def test_chain_analysis_reports_the_known_scale(tmp_path):
    """The accepted chain is 680 resources / 4.39 GB and 234 stages."""
    chain = (Path.home() / 'AppData/Local/Temp/opencode' /
             'm09h-complete-chain-20261030/chain/graph.txt')
    if not chain.is_file():
        pytest.skip('accepted chain evidence is not present')
    report = memory.analyse(chain)
    assert report['stages'] == 234 and report['resources'] == 680
    assert report['declared_bytes'] == 4389619168
    assert report['peak_live_bytes'] <= report['declared_bytes']
    assert report['peak_production_bytes'] < report['peak_live_bytes']


def test_baseline_hashes_and_mode_flags_are_the_accepted_ones():
    assert baseline.ACCEPTED == {
        'fp16': 'dc97c4cbe99346111870507735b9c7bc70846f7c7bf163f1c531ddb810d6e95b',
        'int8': 'f76ac032f45a91dc6c4ead22979eff16ed1d04c4a4cf1add2ff1adc0bd4d6c00',
    }
    flags = baseline.MODES['int8']
    assert flags[flags.index('--cumulative-mode') + 1] == 'spatial'
    assert '--attention-mode' in flags and '--enc0-mode' in flags
    assert baseline.MODES['fp16'] == []


class _Done:
    returncode = 0
    stdout = ''
    stderr = ''


def test_b6_vram_flag_is_opt_in(monkeypatch, tmp_path):
    calls = []
    import m_b6_frame as b6

    def fake_run(command, **kwargs):
        calls.append(command)
        Path(command[command.index('--out') + 1]).mkdir(parents=True, exist_ok=True)
        return _Done()

    monkeypatch.setattr(b6.subprocess, 'run', fake_run)
    b6.run('probe.exe', tmp_path / 'stage.txt', tmp_path / 'out', tmp_path,
           timing=None, vram=None)
    b6.run('probe.exe', tmp_path / 'stage.txt', tmp_path / 'out2', tmp_path,
           timing=tmp_path / 't.json', vram=tmp_path / 'v.json')
    assert '--vram' not in calls[0] and '--timing' not in calls[0]
    assert calls[1][calls[1].index('--vram') + 1] == str(tmp_path / 'v.json')
    assert calls[1][calls[1].index('--timing') + 1] == str(tmp_path / 't.json')
