"""M05 (B4/C19) contract tests: image metadata, resource association, integrity.

The capture extension writes `image` / `image-bind` / `image-unknown` lines and
`image-<id>.bin` files next to the buffer dumps.  These tests build a synthetic
capture directory - no GPU, no NVIDIA data - and check that the audit reads it
correctly:

  * metadata (dimensions, format, channels, row pitch, byte extent) is parsed as
    written, including the alias case where one array is bound twice;
  * a role is decided by content (changed or not), not by an object's type;
  * an edge is confirmed only when the producer's post-launch bytes equal the
    consumer's pre-launch bytes;
  * an image with no in-frame producer is reported as external, not verified;
  * a declared image with no file is reported as missing, never silently dropped.

The oracles are the bytes this test writes itself.

Run: python -m unittest tests.test_m_capture_image_audit -v
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

import m_capture_audit as audit  # noqa: E402


def write_image(stage, image_id, data, post=None):
    os.makedirs(stage, exist_ok=True)
    np.frombuffer(bytes(data), np.uint8).tofile(os.path.join(stage, f'image-{image_id}.bin'))
    if post is not None:
        os.makedirs(os.path.join(stage, 'gpu-output'), exist_ok=True)
        np.frombuffer(bytes(post), np.uint8).tofile(
            os.path.join(stage, 'gpu-output', f'image-{image_id}.bin'))


def write_manifest(stage, lines):
    os.makedirs(stage, exist_ok=True)
    with open(os.path.join(stage, 'manifest.txt'), 'w') as handle:
        handle.write('\n'.join(lines) + '\n')


def image_line(index, kind, handle, array, width, height, depth, fmt, channels,
               original, row, total, pitch=0):
    return (f'image {index} {kind} 0x{handle:x} 0x{array:x} 0x0 {width} {height} '
            f'{depth} 0x{fmt:02x} {channels} {original} {row} {total} {pitch}')


class Metadata(unittest.TestCase):
    def test_alias_and_unknown_bindings_are_reported(self):
        stage = os.path.join(self.tmp, 'replay-000001-rrlite_dec0_4x4_folded')
        write_manifest(stage, [
            'kernel rrlite_dec0_4x4_folded',
            'launch 120 68 1 32 1 1 0',
            'args 168',
            image_line(0, 'surf', 0x30800d000, 0x1b3642d3390, 960, 540, 1, 0x08, 4,
                       200, 3840, 2073600),
            'image-bind 0 56 surf',
            image_line(1, 'tex', 0x308006000, 0x1b3642d3390, 960, 540, 1, 0x08, 4,
                       200, 3840, 2073600),
            'image-bind 1 64 tex',
            'image-unknown 128 0x3006ff000 mipmapped resource has no readback path',
        ])
        table, unknown = audit.images(stage)
        self.assertEqual(sorted(table), [0, 1])
        # Two handles, one backing array: the target is one allocation, not two.
        self.assertEqual(table[0]['array'], table[1]['array'])
        self.assertEqual(len(table[0]['bindings']), 1)
        self.assertEqual(table[0]['bindings'][0]['offset'], 56)
        self.assertEqual(table[1]['bindings'][0]['kind'], 'tex')
        self.assertEqual(table[0]['row_bytes'], 3840)
        self.assertEqual(table[0]['bytes'], 2073600)
        self.assertEqual(table[0]['original'], 200)
        self.assertEqual(len(unknown), 1)
        self.assertIn('mipmapped', unknown[0]['reason'])

    def test_state_distinguishes_written_unchanged_and_missing(self):
        stage = os.path.join(self.tmp, 'stage')
        write_image(stage, 0, bytes(range(32)), post=bytes(range(32)))
        write_image(stage, 1, bytes(32), post=b'\x01' + bytes(31))
        write_image(stage, 2, bytes(32))
        self.assertFalse(audit.image_state(stage, 0)['wrote'])
        state = audit.image_state(stage, 1)
        self.assertTrue(state['wrote'])
        self.assertEqual(state['changed_bytes'], 1)
        self.assertEqual((state['first'], state['last']), (0, 0))
        self.assertIsNone(audit.image_state(stage, 2))

    def setUp(self):
        import tempfile
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = self._dir.name

    def tearDown(self):
        self._dir.cleanup()


class Edges(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._dir = tempfile.TemporaryDirectory()
        self.root = self._dir.name

    def tearDown(self):
        self._dir.cleanup()

    def stage(self, name, rank, instance, images):
        directory = os.path.join(self.root, name)
        write_manifest(directory, ['kernel x', 'launch 1 1 1 1 1 1 0', 'args 8'])
        row = {'stage': name.split('-')[0], 'instance': instance, 'rank': rank,
               'directory': directory, 'images': {}, 'image_unknown': []}
        for image_id, (array, pre, post, kind) in images.items():
            write_image(directory, image_id, pre, post=post)
            row['images'][image_id] = {
                'kind': kind, 'array': array, 'pitched': 0, 'width': 4, 'height': 2,
                'depth': 1, 'format': 0x10, 'channels': 4, 'original': 0,
                'row_bytes': 32, 'bytes': 64, 'pitch': 0,
                'bindings': [{'image': image_id, 'offset': 24, 'kind': kind}],
            }
        return row

    def test_content_verified_edge(self):
        pattern = bytes(range(64))
        producer = self.stage('dec0-1', 14, 1, {0: (0xAA, bytes(64), pattern, 'surf')})
        consumer = self.stage('post-1', 15, 1, {0: (0xAA, pattern, pattern, 'tex')})
        edges, missing = audit.image_edges([producer, consumer])
        self.assertFalse(missing)
        self.assertEqual(len(edges), 1)
        self.assertTrue(edges[0]['verified'])
        self.assertEqual(edges[0]['edge'], 'dec0#1 -> post#1')
        self.assertEqual(edges[0]['differing'], 0)

    def test_same_array_with_different_bytes_is_not_an_edge(self):
        producer = self.stage('dec0-1', 14, 1, {0: (0xAA, bytes(64), bytes(range(64)), 'surf')})
        consumer = self.stage('post-1', 15, 1, {0: (0xAA, bytes(64), bytes(64), 'tex')})
        edges, _ = audit.image_edges([producer, consumer])
        self.assertFalse(edges[0]['verified'])
        self.assertEqual(edges[0]['differing'], 63)

    def test_input_without_in_frame_producer_is_external(self):
        consumer = self.stage('enc0-1', 0, 1, {0: (0xBB, bytes(range(64)), bytes(range(64)), 'tex')})
        edges, _ = audit.image_edges([consumer])
        self.assertEqual(len(edges), 1)
        self.assertIsNone(edges[0]['verified'])
        self.assertIn('earlier frame', edges[0]['reason'])

    def test_producer_must_come_first(self):
        # Same array, but the writing stage is later in the frame: not an edge.
        early = self.stage('enc0-1', 0, 1, {0: (0xCC, bytes(64), bytes(64), 'tex')})
        late = self.stage('dec0-1', 14, 1, {0: (0xCC, bytes(64), bytes(range(64)), 'surf')})
        edges, _ = audit.image_edges([early, late])
        self.assertIsNone(edges[0]['verified'])

    def test_missing_capture_is_reported(self):
        producer = self.stage('dec0-1', 14, 1, {0: (0xDD, bytes(64), bytes(64), 'surf')})
        os.remove(os.path.join(producer['directory'], 'gpu-output', 'image-0.bin'))
        edges, missing = audit.image_edges([producer])
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]['stage'], 'dec0')
        self.assertEqual(edges, [])


if __name__ == '__main__':
    unittest.main()
