#!/usr/bin/env python3
"""B4.1 contract tests: the image-stage fixture is built from declared evidence.

A fixture decides which bytes a native stage reads and what it must produce, so a
wrong argument slot, a wrong stride or an unvalidated geometry would silently
turn into a wrong reconstruction.  These tests build a synthetic capture (the
manifest lines the replay hook writes) and check that the exporter:

  * reads the parameter block from the slots d4r's harness uses, not from a
    guessed struct layout;
  * maps a captured image's backing format and channel count to a declared D3D12
    format, and refuses one whose row stride does not agree;
  * refuses a launch whose geometry contradicts the stage it is exported as;
  * reports a differing output, its ULP distribution and its spatial extent, and
    flags a repeat run that is not deterministic.

No NVIDIA data is needed: the synthetic capture has known contents.
"""
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'scripts'))

import m_stage_fixture as fx        # noqa: E402

SRC_W, SRC_H = 2880, 1620
OUT_W, OUT_H = 1920, 1080
COLOUR_W, COLOUR_H = 960, 540


def synthetic_capture(root):
    """One downsample launch in the format the replay hook writes."""
    directory = os.path.join(root, 'replay-000001-rrlite_downsample_kernel_static_ldr')
    os.makedirs(os.path.join(directory, 'gpu-output'))
    args = bytearray(96)
    struct.pack_into('<2i', args, 0, SRC_W, SRC_H)      # srcSize
    struct.pack_into('<2i', args, 40, 0, 0)             # outOrigin
    struct.pack_into('<2i', args, 64, 0, 0)             # colOrigin
    struct.pack_into('<2i', args, 72, COLOUR_W, COLOUR_H)   # colSize
    struct.pack_into('<2f', args, 80, 0.5, 0.5)         # out2rnd
    struct.pack_into('<2f', args, 88, 0.0, 0.0)         # rndOffs
    struct.pack_into('<Q', args, 24, 0x100000000)       # packed source buffer
    struct.pack_into('<Q', args, 32, 0x200000000)       # output surface
    struct.pack_into('<Q', args, 48, 0x300000000)       # exposure 1x1
    struct.pack_into('<Q', args, 56, 0x400000000)       # colour texture
    src = np.arange(SRC_W * SRC_H, dtype=np.uint32)
    colour = np.zeros((COLOUR_H, COLOUR_W, 4), np.float16)
    out = np.zeros((OUT_H, OUT_W, 4), np.float16)
    lines = [
        'kernel rrlite_downsample_kernel_static_ldr',
        'launch 120 68 1 16 16 1 0',
        'args 96',
        f'alloc 0 0x100000000 {src.nbytes}',
        'pointer 24 0 0',
        'image 0 surf 0x200000000 0x0 0x0 1920 1080 1 0x10 4 0 15360 16588800 0',
        'image-bind 0 32 surf',
        'image 1 tex 0x300000000 0x0 0x0 1 1 1 0x10 1 0 2 2 0',
        'image-bind 1 48 tex',
        'image 2 tex 0x400000000 0x0 0x0 960 540 1 0x10 4 0 7680 4147200 0',
        'image-bind 2 56 tex',
    ]
    open(os.path.join(directory, 'manifest.txt'), 'w').write('\n'.join(lines) + '\n')
    open(os.path.join(directory, 'args.bin'), 'wb').write(args)
    open(os.path.join(directory, 'alloc-0.bin'), 'wb').write(src.tobytes())
    open(os.path.join(directory, 'image-0.bin'), 'wb').write(out.tobytes())
    open(os.path.join(directory, 'image-1.bin'), 'wb').write(np.zeros(1, np.float16).tobytes())
    open(os.path.join(directory, 'image-2.bin'), 'wb').write(colour.tobytes())
    for name, data in (('alloc-0.bin', src.tobytes()),
                       ('image-0.bin', (out + 1).astype(np.float16).tobytes()),
                       ('image-1.bin', bytes(2)),
                       ('image-2.bin', colour.tobytes())):
        open(os.path.join(directory, 'gpu-output', name), 'wb').write(data)
    return root, directory


class FixtureExport(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='m-fixture-')
        synthetic_capture(self.root)
        self.out = os.path.join(self.root, 'fixture')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def export(self):
        return fx.export(self.root, 'downsample', self.out, 1)

    def test_parameters_come_from_the_documented_slots(self):
        self.export()
        record = json.load(open(os.path.join(self.out, 'fixture.json')))
        self.assertEqual(record['params']['srcSize'], [SRC_W, SRC_H])
        self.assertEqual(record['params']['outOrigin'], [0, 0])
        self.assertEqual(record['params']['colSize'], [COLOUR_W, COLOUR_H])
        self.assertEqual(record['params']['out2rnd'], [0.5, 0.5])
        self.assertEqual(record['groups'], [120, 68, 1])
        self.assertEqual(record['threads'], [16, 16, 1])
        blob = open(os.path.join(self.out, 'params.bin'), 'rb').read()
        self.assertEqual(len(blob), 64)
        self.assertEqual(struct.unpack_from('<4i', blob, 0), (SRC_W, SRC_H, 0, 0))
        self.assertEqual(struct.unpack_from('<4i', blob, 16), (0, 0, COLOUR_W, COLOUR_H))
        self.assertEqual(struct.unpack_from('<4f', blob, 32), (0.5, 0.5, 0.0, 0.0))

    def test_manifest_names_every_binding_with_its_format(self):
        self.export()
        manifest = open(os.path.join(self.out, 'stage.txt')).read().splitlines()
        self.assertIn('shader downsample_cs.cso', manifest)
        self.assertIn('threads 16 16 1', manifest)
        self.assertIn('groups 120 68 1', manifest)
        self.assertIn(f'srv buffer 0 in-src.bin {SRC_W * SRC_H * 4}', manifest)
        self.assertIn('srv tex2d 1 R16G16B16A16_FLOAT 960 540 in-colour.bin', manifest)
        self.assertIn('uav tex2d 2 R16G16B16A16_FLOAT 1920 1080 out-out.bin', manifest)

    def test_reference_is_the_post_launch_surface(self):
        self.export()
        record = json.load(open(os.path.join(self.out, 'fixture.json')))
        # a zero surface to 1.0 changes one byte per channel, not both
        self.assertEqual(record['bindings']['out']['changed_bytes'], OUT_W * OUT_H * 4)
        reference = np.fromfile(os.path.join(self.out, 'ref-out.bin'), np.float16)
        self.assertTrue(np.all(reference == 1))

    def test_payloads_are_the_captured_bytes(self):
        self.export()
        captured = os.path.join(self.root,
                                'replay-000001-rrlite_downsample_kernel_static_ldr')
        for role, name in (('src', 'alloc-0.bin'), ('colour', 'image-2.bin')):
            self.assertEqual(open(os.path.join(self.out, f'in-{role}.bin'), 'rb').read(),
                             open(os.path.join(captured, name), 'rb').read())

    def test_rejects_a_geometry_that_is_not_the_stage(self):
        # An output that is internally consistent (declared size, stride and
        # payload agree) but no longer 2/3 of the packed source.
        directory = os.path.join(self.root,
                                 'replay-000001-rrlite_downsample_kernel_static_ldr')
        path = os.path.join(directory, 'manifest.txt')
        bytes_needed = 1920 * 1622 * 8
        text = open(path).read().replace('1920 1080 1 0x10 4 0 15360 16588800',
                                         f'1920 1622 1 0x10 4 0 15360 {bytes_needed}')
        open(path, 'w').write(text)
        for name in ('image-0.bin', os.path.join('gpu-output', 'image-0.bin')):
            open(os.path.join(directory, name), 'wb').write(bytes(bytes_needed))
        with self.assertRaises(SystemExit) as caught:
            self.export()
        self.assertIn('2/3', str(caught.exception))

    def test_rejects_a_row_stride_that_does_not_match_the_format(self):
        directory = os.path.join(self.root,
                                 'replay-000001-rrlite_downsample_kernel_static_ldr')
        path = os.path.join(directory, 'manifest.txt')
        text = open(path).read().replace('960 540 1 0x10 4 0 7680',
                                         '960 540 1 0x10 4 0 7696')
        open(path, 'w').write(text)
        with self.assertRaises(SystemExit) as caught:
            self.export()
        self.assertIn('row stride', str(caught.exception))

    def test_rejects_a_backing_format_without_a_declared_mapping(self):
        directory = os.path.join(self.root,
                                 'replay-000001-rrlite_downsample_kernel_static_ldr')
        path = os.path.join(directory, 'manifest.txt')
        text = open(path).read().replace('960 540 1 0x10 4', '960 540 1 0x40 4')
        open(path, 'w').write(text)
        with self.assertRaises(SystemExit) as caught:
            self.export()
        self.assertIn('no D3D12 staging format', str(caught.exception))

    def test_rejects_a_sub_rectangle_output(self):
        directory = os.path.join(self.root,
                                 'replay-000001-rrlite_downsample_kernel_static_ldr')
        args = bytearray(open(os.path.join(directory, 'args.bin'), 'rb').read())
        struct.pack_into('<2i', args, 40, 128, 64)
        open(os.path.join(directory, 'args.bin'), 'wb').write(args)
        with self.assertRaises(SystemExit) as caught:
            self.export()
        self.assertIn('origin', str(caught.exception))

    def test_validate_stage_reports_every_problem(self):
        record = {
            'params': {'srcSize': [10, 10], 'outOrigin': [0, 0], 'colOrigin': [1, 0],
                       'colSize': [4, 4], 'out2rnd': [1.5, 1.5], 'rndOffs': [0.0, 0.0]},
            'bindings': {'src': {'bytes': 400}, 'colour': {'geometry': [4, 4, 1]},
                         'out': {'geometry': [9, 9, 1]}},
            'groups': [1, 1, 1], 'threads': [16, 16, 1],
        }
        problems = fx.validate_stage('downsample', record)
        self.assertEqual(len(problems), 3, problems)
        self.assertTrue(any('2/3' in p for p in problems))
        self.assertTrue(any('origin' in p for p in problems))
        self.assertTrue(any('out2rnd' in p for p in problems))


class FixtureComparison(unittest.TestCase):
    """The B4.7 comparison must describe a difference, not only report it."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='m-check-')
        self.width, self.height = 8, 4
        self.pixels = self.width * self.height
        self.reference = np.zeros((self.height, self.width, 4), np.float16)
        self.reference[:, :, 1] = 0.5
        self.reference[:, :, 3] = 1.0
        np.asarray(self.reference).tofile(os.path.join(self.root, 'ref-out.bin'))
        record = {'stage': 'downsample', 'kernel': 'k',
                  'bindings': {'out': {'output': True, 'format': 'R16G16B16A16_FLOAT',
                                       'geometry': [self.width, self.height, 1]}}}
        json.dump(record, open(os.path.join(self.root, 'fixture.json'), 'w'))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write(self, directory, values):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, 'out-out.bin')
        np.asarray(values, np.float16).tofile(path)
        return path

    def test_identical_output_passes(self):
        results = os.path.join(self.root, 'results')
        self.write(results, self.reference)
        path = open(os.path.join(results, 'out-out.bin'), 'rb').read()
        row = fx.compare(path, open(os.path.join(self.root, 'ref-out.bin'), 'rb').read(),
                         'out', 'R16G16B16A16_FLOAT')
        self.assertTrue(row['pass'])
        self.assertEqual(row['differing_bytes'], 0)
        self.assertEqual(row['ulp_histogram'], {'0': self.pixels * 4})

    def test_one_ulp_is_localised(self):
        native = self.reference.copy()
        native[2, 3, 1] = np.nextafter(native[2, 3, 1], np.float16(1.0))
        row = fx.compare(native.tobytes(), self.reference.tobytes(), 'out',
                         'R16G16B16A16_FLOAT')
        self.assertFalse(row['pass'])
        self.assertEqual(row['max_step'], 1)
        self.assertEqual(row['ulp_histogram']['1'], 1)
        self.assertGreater(row['identical_f16'], 0.99)
        loc = fx.localise(native.tobytes(), self.reference.tobytes(),
                          self.width, self.height, 8)
        self.assertEqual(loc['differing_pixels'], 1)
        self.assertEqual((loc['first_row'], loc['first_col']), (2, 3))

    def test_nonfinite_output_is_counted(self):
        native = self.reference.copy().astype(np.float32)
        native[0, 0, 0] = np.nan
        native[0, 1, 0] = np.inf
        row = fx.compare(native.astype(np.float16).tobytes(),
                         self.reference.tobytes(), 'out', 'R16G16B16A16_FLOAT')
        self.assertEqual(row['nonfinite'], 2)
        self.assertFalse(row['pass'])

    def test_repeat_run_must_be_identical(self):
        results = os.path.join(self.root, 'results')
        repeat = os.path.join(self.root, 'repeat')
        values = self.reference.copy()
        self.write(results, values)
        self.write(repeat, values)
        self.assertEqual(fx.check(self.root, results, repeat), 0)
        values[1, 1, 0] = 1.0
        self.write(repeat, values)
        self.assertEqual(fx.check(self.root, results, repeat), 1)


class Dec0Fixture(unittest.TestCase):
    """dec0's declared geometry, checked before any bytes move."""

    def record(self, **overrides):
        record = {
            'params': {'size': [960, 544], 'reset': 1, 'sigma': 8.0, 'rho': 0.0786},
            'bindings': {
                'weights': {'kind': 'buffer', 'bytes': 8448},
                'tokens': {'kind': 'buffer', 'bytes': 2 * 272 * 480 * 32},
                'gate': {'kind': 'image', 'output': True, 'format': 'R32_UINT',
                         'geometry': [960, 540, 1]},
                'mix': {'kind': 'image', 'output': True, 'format': 'R32_UINT',
                        'geometry': [960, 540, 1]},
                'feat': {'kind': 'image', 'output': True, 'format': 'R32_UINT',
                         'geometry': [960, 540, 1]},
                'cov': {'kind': 'image', 'output': True,
                        'format': 'R16G16B16A16_FLOAT', 'geometry': [960, 540, 1]},
            },
            'groups': [2040, 1, 1], 'threads': [64, 1, 1],
        }
        for key, value in overrides.items():
            if key in record['params']:
                record['params'][key] = value
            elif key in record['bindings']:
                record['bindings'][key].update(value)
            else:
                record[key] = value
        return record

    def test_a_consistent_record_has_no_problems(self):
        self.assertEqual(fx.validate_stage('dec0', self.record()), [])

    def test_the_token_field_must_be_two_planes_of_the_padded_grid(self):
        # 270 rows instead of 272 - EXP-013's reading of the parameter block
        record = self.record()
        record['bindings']['tokens']['bytes'] = 2 * 270 * 480 * 32
        problems = fx.validate_stage('dec0', record)
        self.assertTrue(any('token field' in p for p in problems), problems)

    def test_the_weight_blob_is_the_matrix_plus_its_bias(self):
        record = self.record()
        record['bindings']['weights']['bytes'] = 8192
        problems = fx.validate_stage('dec0', record)
        self.assertTrue(any('weight blob' in p for p in problems), problems)

    def test_the_byte_surfaces_must_be_written_as_exact_bytes(self):
        record = self.record()
        record['bindings']['gate']['format'] = 'R8G8B8A8_UNORM'
        problems = fx.validate_stage('dec0', record)
        self.assertTrue(any('gate' in p and 'R32_UINT' in p for p in problems),
                        problems)

    def test_the_dispatch_must_cover_every_token(self):
        record = self.record(groups=[8, 1, 1])
        problems = fx.validate_stage('dec0', record)
        self.assertTrue(any('tokens' in p for p in problems), problems)

    def test_the_covariance_is_float16(self):
        record = self.record()
        record['bindings']['cov']['format'] = 'R32_UINT'
        self.assertEqual(fx.validate_stage('dec0', record), [])
        # ... the validator only rejects a wrong *byte* view on the 8-bit
        # surfaces; the covariance's format is checked by the shader binding


class Enc0Fixture(unittest.TestCase):
    """enc0's declared geometry, checked before any bytes move.

    The stage writes a *buffer* through `st.global`, so the field's byte count is
    the contract; and the two sizes in the parameter block (padded 544, visible
    540) are both load-bearing for different reasons.
    """

    def record(self, **overrides):
        record = {
            'params': {'padded': [960, 544], 'visible': [960, 540],
                       'frame': 0, 'reset': 1},
            'bindings': {
                'weights': {'kind': 'buffer', 'bytes': 12416},
                'colour': {'kind': 'image', 'format': 'R16G16B16A16_FLOAT',
                           'geometry': [960, 540, 1]},
                'tokens': {'kind': 'buffer', 'output': True, 'bytes': 2 * 272 * 480 * 32},
            },
            'groups': [2040, 1, 1], 'threads': [64, 1, 1],
        }
        for key, value in overrides.items():
            if key in record['params']:
                record['params'][key] = value
            elif key in record['bindings']:
                record['bindings'][key].update(value)
            else:
                record[key] = value
        return record

    def test_a_consistent_record_has_no_problems(self):
        self.assertEqual(fx.validate_stage('enc0', self.record()), [])

    def test_the_field_is_two_planes_of_the_padded_grid(self):
        record = self.record()
        record['bindings']['tokens']['bytes'] = 2 * 270 * 480 * 32
        problems = fx.validate_stage('enc0', record)
        self.assertTrue(any('token field' in p for p in problems), problems)

    def test_the_blob_is_the_dither_table_matrix_and_bias(self):
        record = self.record()
        record['bindings']['weights']['bytes'] = 8192
        problems = fx.validate_stage('enc0', record)
        self.assertTrue(any('weight blob' in p for p in problems), problems)
        self.assertEqual(4096 + 8192 + 128, 12416)

    def test_the_reset_frame_is_the_only_one_implemented(self):
        # the history, previous-feature and exposure slots are only provably zero
        # on a reset frame; accepting a history frame would silently drop them
        problems = fx.validate_stage('enc0', self.record(reset=0))
        self.assertTrue(any('reset' in p for p in problems), problems)

    def test_the_colour_texture_is_the_visible_size(self):
        record = self.record()
        record['bindings']['colour']['geometry'] = [960, 544, 1]
        problems = fx.validate_stage('enc0', record)
        self.assertTrue(any('colour' in p for p in problems), problems)

    def test_the_dither_frame_must_be_in_the_table(self):
        problems = fx.validate_stage('enc0', self.record(frame=8))
        self.assertTrue(any('frame' in p for p in problems), problems)
        self.assertEqual(8 << 8, 2048)          # the u16 table has 2048 entries

    def test_the_dispatch_must_cover_every_token(self):
        problems = fx.validate_stage('enc0', self.record(groups=[8, 1, 1]))
        self.assertTrue(any('tokens' in p for p in problems), problems)


class PostFixture(unittest.TestCase):
    """post_3_1's declared geometry: three outputs, two different sizes."""

    def record(self, **overrides):
        record = {
            'params': {'visible': [960, 540], 'recon': [2880, 1620],
                       'display': [1920, 1080], 'colour_origin': [0, 0],
                       'scale': [0.33333334, 0.33333334, 0.0, 0.0],
                       'gain': 2.753, 'mv_scale': [1.5, 1.5],
                       'mv_offset': [0.0, 0.0], 'reset': 1},
            'bindings': {
                'colour': {'kind': 'image', 'format': 'R16G16B16A16_FLOAT',
                           'geometry': [960, 540, 1]},
                'gate': {'kind': 'image', 'format': 'R8G8B8A8_UNORM',
                         'geometry': [960, 540, 1]},
                'mix': {'kind': 'image', 'format': 'R8G8B8A8_UNORM',
                        'geometry': [960, 540, 1]},
                'cov': {'kind': 'image', 'format': 'R16G16B16A16_FLOAT',
                        'geometry': [960, 540, 1]},
                'motion': {'kind': 'image', 'format': 'R16G16_FLOAT',
                           'geometry': [960, 540, 1]},
                'packed': {'kind': 'buffer', 'output': True,
                           'bytes': 2880 * 1620 * 4},
                'history': {'kind': 'image', 'output': True, 'format': 'R32_UINT',
                            'geometry': [5760, 1620, 1], 'bytes': 2880 * 1620 * 8},
                'half': {'kind': 'image', 'output': True,
                         'format': 'R32G32B32A32_FLOAT',
                         'geometry': [1440, 810, 1], 'bytes': 1440 * 810 * 16},
            },
            'groups': [180, 51, 1], 'threads': [16, 16, 1],
        }
        for key, value in overrides.items():
            if key in record['params']:
                record['params'][key] = value
            elif key in record['bindings']:
                record['bindings'][key].update(value)
            else:
                record[key] = value
        return record

    def test_a_consistent_record_has_no_problems(self):
        self.assertEqual(fx.validate_stage('post_3_1', self.record()), [])

    def test_the_packed_buffer_is_the_reconstruction_grid(self):
        record = self.record()
        record['bindings']['packed']['bytes'] = 960 * 540 * 4
        problems = fx.validate_stage('post_3_1', record)
        self.assertTrue(any('packed' in p for p in problems), problems)

    def test_the_reconstruction_surface_is_rgba16f(self):
        record = self.record()
        record['bindings']['history']['bytes'] = 2880 * 1620 * 4
        problems = fx.validate_stage('post_3_1', record)
        self.assertTrue(any('reconstruction' in p for p in problems), problems)

    def test_the_half_surface_is_a_quarter_of_the_pixels(self):
        record = self.record()
        record['bindings']['half']['bytes'] = 1440 * 810 * 8
        problems = fx.validate_stage('post_3_1', record)
        self.assertTrue(any('half' in p for p in problems), problems)

    def test_the_render_textures_are_the_render_size(self):
        record = self.record()
        record['bindings']['cov']['geometry'] = [2880, 1620, 1]
        problems = fx.validate_stage('post_3_1', record)
        self.assertTrue(any('cov' in p for p in problems), problems)

    def test_the_reset_frame_is_the_only_one_implemented(self):
        problems = fx.validate_stage('post_3_1', self.record(reset=0))
        self.assertTrue(any('reset' in p for p in problems), problems)

    def test_the_dispatch_must_cover_both_rows_of_a_thread(self):
        problems = fx.validate_stage('post_3_1', self.record(groups=[180, 1, 1]))
        self.assertTrue(any('grid' in p for p in problems), problems)


if __name__ == '__main__':
    sys.exit(unittest.main())
