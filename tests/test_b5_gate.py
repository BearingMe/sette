"""B5 acceptance logic: fail closed, and never accept a stale or absent reference."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from m_conformance import B5, B5_LAYERS, judge_b5_chain, judge_b5_layers


def row(**overrides):
    base = {'bytes': 100, 'exact': 0.96, 'differing_codes': 4, 'nonfinite_pairs': 0,
            'signed_mean_decoded': 0.0, 'saturated_reference': 0, 'saturated_actual': 0,
            'relative_rms': 0.01, 'max_abs_decoded': 0.5}
    base.update(overrides)
    return base


def summary(layer_overrides=None, chain_overrides=None, keep_chain=True):
    layers = {}
    for label in B5_LAYERS:
        pair = {'output': row()}
        if label in ('enc1', 'enc2'):
            pair['merge'] = row()
        pair.update((layer_overrides or {}).get(label, {}))
        layers[label] = {'status': 'pass',
                         'deterministic': {'fast': True, 'conservative': True,
                                           'sette': True},
                         'pairs': {'sette-vs-translated': pair}}
    build = {'layers': layers}
    if keep_chain:
        chain = {}
        for label in B5_LAYERS:
            pairs = {'translated-vs-sette': {'output': row(exact=0.215728)},
                     'canonical-vs-translated': {'output': row(exact=0.217096)}}
            pairs.update((chain_overrides or {}).get(label, {}))
            chain[label] = {'translated_dispatches': 1, 'native_dispatches': 0,
                            'pairs': pairs}
        build['chain'] = chain
    return build


def test_complete_reference_satisfies_the_declared_criteria():
    build = summary()
    assert judge_b5_layers(build)[1] == []
    assert judge_b5_chain(build)[1] == []
    assert len(judge_b5_layers(build)[0]) == 12
    assert [row['boundary'] for row in judge_b5_chain(build)[0]] == B5_LAYERS


def test_the_floor_cannot_exceed_the_reference_pair_agreement():
    # The original PTX and the native replacement agree on only 94.59% of codes on
    # the hardest layer; a floor at or above that would fail an independent
    # implementation, so the declared floor must stay below it.
    assert B5['layer_identical_min'] < 0.9459


@pytest.mark.parametrize('kind,value', [('exact', 0.85), ('relative_rms', 0.03),
                                        ('max_abs_decoded', 2.0),
                                        ('signed_mean_decoded', 0.002),
                                        ('nonfinite_pairs', 3),
                                        ('saturated_actual', 7)])
def test_any_layer_bound_violation_fails(kind, value):
    build = summary(layer_overrides={'tube#4': {'output': row(**{kind: value})}})
    problems = judge_b5_layers(build)[1]
    assert any('tube#4.output' in problem for problem in problems)


def test_a_missing_layer_fails_instead_of_being_skipped():
    build = summary()
    del build['layers']['dec2']
    assert any('dec2' in problem and 'no completed' in problem
               for problem in judge_b5_layers(build)[1])


def test_a_nondeterministic_layer_fails():
    build = summary()
    build['layers']['enc2']['deterministic']['sette'] = False
    assert any('not deterministic' in problem for problem in judge_b5_layers(build)[1])


def test_an_empty_or_absent_reference_fails():
    assert judge_b5_layers({})[1]
    assert judge_b5_chain({})[1]
    assert judge_b5_chain({'layers': {}})[1]


def test_the_chain_gate_is_relative_to_the_reference_pair():
    within = summary(chain_overrides={'dec1': {
        'translated-vs-sette': {'output': row(exact=0.217096 - B5['chain_identical_margin'])}}})
    assert judge_b5_chain(within)[1] == []
    beyond = summary(chain_overrides={'dec1': {
        'translated-vs-sette': {'output': row(exact=0.15)}}})
    assert any('below the reference' in problem for problem in judge_b5_chain(beyond)[1])


def test_chain_magnitude_and_provenance_are_checked():
    worse = summary(chain_overrides={'dec1': {
        'translated-vs-sette': {'output': row(relative_rms=0.2, max_abs_decoded=9.0)}}})
    problems = judge_b5_chain(worse)[1]
    assert any('relative RMS' in problem for problem in problems)
    assert any('maximum decoded deviation' in problem for problem in problems)
    stale = summary()
    stale['chain']['enc1']['translated_dispatches'] = 0
    stale['chain']['enc2']['native_dispatches'] = 1
    problems = judge_b5_chain(stale)[1]
    assert any('no translated dispatch' in problem for problem in problems)
    assert any('native dispatches' in problem for problem in problems)


def test_legacy_gate_results_do_not_veto_the_translated_criterion():
    build = summary()
    build['legacy'] = {'pass': False, 'chain': {'problems': ['below floor']}}
    assert judge_b5_layers(build)[1] == []
    assert judge_b5_chain(build)[1] == []
