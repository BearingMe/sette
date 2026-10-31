"""M15-A: lifetime classification, arena planning and production-mode wiring.

The native host is authoritative; these tests lock the pure planning rules and
the accepted chain's plan, and verify that the Python inventory agrees with
native ``--arena-report`` evidence when it is present.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import m15_arena as arena

TEMP = Path.home() / 'AppData/Local/Temp/opencode'
CHAIN_CANDIDATES = [
    TEMP / 'm15a-measure2/chain/graph.txt',
    TEMP / 'm14-production-chain3/graph.txt',
]

GRAPH = """version 1
resource weights 100 weights.bin
resource input 100 input.bin
resource a 100
resource b 100
resource c 100
resource d 100
resource pre 100
resource both 100
resource scratch 100
resource unused 100
stage s0
shader x.cso
threads 1 1 1
groups 1 1 1
params p 16
srv 0 weights
srv 1 input
srv 2 pre
srv 3 both
uav 0 a
uav 1 both
stage s1
shader y.cso
threads 1 1 1
groups 1 1 1
params p 16
srv 0 a
uav 0 b
uav 1 scratch
stage s2
shader z.cso
threads 1 1 1
groups 1 1 1
params p 16
srv 0 b
uav 0 c
stage s3
shader w.cso
threads 1 1 1
groups 1 1 1
params p 16
srv 0 c
uav 0 d
dump c 0 100 out-c.bin
dump d 0 100 out-d.bin
"""


def chain_graph():
    for path in CHAIN_CANDIDATES:
        if path.is_file():
            return path
    pytest.skip('accepted chain graph is not present')


def write(tmp_path, text=GRAPH):
    path = tmp_path / 'graph.txt'
    path.write_text(text)
    return path


def test_first_writer_last_reader_and_initialization(tmp_path):
    report = arena.analyse(write(tmp_path))
    resources = {r['name']: r for r in report['resources']}
    assert resources['a']['first_write'] == 0
    assert resources['a']['last_read'] == 1
    assert resources['weights']['lifetime'] == 'persistent'
    assert resources['input']['lifetime'] == 'persistent'
    assert resources['unused']['lifetime'] == 'dead'
    assert resources['scratch']['diagnostic_only'] is True


def test_read_before_write_and_dual_state_resources_are_dedicated(tmp_path):
    report = arena.analyse(write(tmp_path))
    resources = {r['name']: r for r in report['resources']}
    assert resources['pre']['reads_before_write'] is True
    assert resources['pre']['lifetime'] == 'dedicated'
    assert resources['both']['both_states_one_stage'] is True
    assert resources['both']['lifetime'] == 'dedicated'


def test_persistent_and_transient_bytes_and_peak(tmp_path):
    report = arena.analyse(write(tmp_path))
    # weights/input are persistent; a,b,c,d share one 64 KiB slot.
    assert report['arena']['persistent_bytes'] == 2 * 65536
    assert report['arena']['heap_bytes'] == 3 * 65536
    assert report['arena']['placed_resources'] == 5
    assert report['arena']['dedicated_resources'] == 2       # pre, both
    assert report['arena']['dead_resources'] == 1            # unused
    # s0: weights+input+pre+both+a; s1: a+b+scratch.
    assert report['arena']['peak_live_bytes'] == 500
    assert report['arena']['peak_transient_bytes'] == 300


def test_production_keeps_only_the_final_stage_dump(tmp_path):
    resources, stages, dumps = arena.memory.parse_graph(write(tmp_path))
    selected = arena.select_production_dumps(stages, dumps)
    assert [d['file'] for d in selected] == ['out-d.bin']
    report = arena.analyse(write(tmp_path), mode='production')
    assert report['dumps'] == 1 and report['dropped_dumps'] == 1
    assert report['arena']['peak_live_bytes'] == 500


def test_accepted_chain_plan_is_locked():
    report = arena.analyse(chain_graph())
    a = report['arena']
    assert report['stage_count'] == 234 and report['resource_count'] == 680
    assert report['declared_bytes'] == 4389619168
    assert a['heap_bytes'] == 437387264
    assert a['persistent_bytes'] == 24379392
    assert a['dead_resources'] == 11
    assert a['placed_resources'] == 516
    assert a['peak_live_bytes'] == 385209616
    assert a['peak_transient_bytes'] == 384890896
    assert a['largest_transient_bytes'] == 139919360
    assert a['largest_transient_name'] == 'enc1_i8_mlp-up_acc'
    production = arena.analyse(chain_graph(), mode='production')
    assert production['dumps'] == 1 and production['dropped_dumps'] == 12
    assert [d['file'] for d in
            arena.select_production_dumps(*arena.memory.parse_graph(chain_graph())[1:])] \
        == ['out-dec1.bin']


def test_aliased_intervals_never_overlap():
    report = arena.analyse(chain_graph())
    transient = [r for r in report['resources'] if r['lifetime'] == 'transient']
    for r in transient:
        assert r['offset'] % 65536 == 0
        assert r['aligned_bytes'] >= r['bytes']
    for i, left in enumerate(transient):
        for right in transient[i + 1:]:
            if left['last'] < right['first'] or right['last'] < left['first']:
                continue
            l0, l1 = left['offset'], left['offset'] + left['aligned_bytes']
            r0, r1 = right['offset'], right['offset'] + right['aligned_bytes']
            assert l1 <= r0 or r1 <= l0, (left['name'], right['name'])


def test_python_plan_matches_native_report_when_available():
    native_path = TEMP / 'm15a-measure/production/run-0/arena.json'
    if not native_path.is_file():
        pytest.skip('native arena report is not present')
    native = json.loads(native_path.read_text())
    report = arena.analyse(chain_graph(), mode='production')
    for key in ('heap_bytes', 'persistent_bytes', 'placed_resources',
                'peak_live_bytes', 'peak_transient_bytes',
                'largest_transient_bytes'):
        assert native['arena'][key] == report['arena'][key], key
    assert native['barriers']['aliasing'] == 234
    assert native['barriers']['uav'] == 234


def test_chain_uses_one_aliasing_barrier_per_stage(tmp_path, monkeypatch):
    import m_b6_frame as b6
    calls = []

    class Done:
        returncode = 0
        stdout = ''
        stderr = ''

    def fake_run(command, **kwargs):
        calls.append(command)
        out = Path(command[command.index('--out') + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / 'out-dec1.bin').write_bytes(b'x')
        return Done()

    monkeypatch.setattr(b6.subprocess, 'run', fake_run)
    b6.run_chain('probe.exe', tmp_path, tmp_path / 'out', tmp_path,
                 timing=None, vram=tmp_path / 'v.json',
                 mode='production', arena=True)
    command = calls[0]
    assert command[command.index('--mode') + 1] == 'production'
    assert '--arena' in command
    assert command[command.index('--arena-report') + 1].endswith('v-arena.json')
