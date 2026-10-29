#!/usr/bin/env python3
"""M05 B4.1/B4.7: turn a captured launch into a native D3D12 image-stage fixture.

A capture records each launch's argument block and the bytes behind every
texture, surface and buffer it binds.  A fixture restates that evidence in a form
`src/native/m_image_stage.cpp` can execute: one input file per binding, a
resolved parameter block, the dispatch the capture recorded, and the captured
output as the reference.  Formats, geometry and byte counts are declared, never
inferred, so a mismatch between the capture and what the shader expects is an
error rather than a silently reinterpreted image.

Subcommands:

  roundtrip CAPTURE --out DIR
      B4.1 gate.  Copy each of a representative set of captured resources
      through D3D12 (upload, dispatch, readback) and require the bytes to come
      back unchanged.  A format or pitch mistake shows up here.

  export CAPTURE --stage NAME --out DIR [--frame N]
      Write the stage's fixture: `stage.txt` (the manifest), `params.bin`,
      `in-*.bin`, `ref-*.bin`, and `fixture.json` with every mapping recorded.

  check DIR [--results DIR] [--repeat DIR] [--json OUT]
      B4.7 comparison: native output against the captured reference, with
      exact-byte agreement, f16 agreement, error distribution, ULP histogram,
      spatial localisation, NaN/Inf checks and repeat-run determinism.

Fixtures hold NVIDIA-derived data.  Keep them in a temporary workspace; nothing
here writes to the repository.

usage:
  m_stage_fixture.py roundtrip CAPTURE --out DIR [--probe EXE] [--shaders DIR]
  m_stage_fixture.py export CAPTURE --stage downsample --out DIR [--frame N]
  m_stage_fixture.py check DIR [--results DIR] [--repeat DIR] [--json OUT]
"""
import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import m_capture_audit as audit      # noqa: E402
import m_image_validate as iv        # noqa: E402

# The driver's backing-storage code and channel count, as the capture records
# them, mapped to the D3D12 format a fixture may declare.  0x10 is the 16-bit
# float family and 0x20 the 32-bit one; 0x08 is SNORM8 and 0x01 UNORM8, whose
# bytes are identical under an UNORM view (a shader encodes SNORM itself).
IMAGE_FORMATS = {
    (0x10, 1): 'R16_FLOAT',
    (0x10, 2): 'R16G16_FLOAT',
    (0x10, 4): 'R16G16B16A16_FLOAT',
    (0x20, 1): 'R32_FLOAT',
    (0x20, 2): 'R32G32_FLOAT',
    (0x20, 4): 'R32G32B32A32_FLOAT',
    (0x08, 4): 'R8G8B8A8_UNORM',
    (0x01, 4): 'R8G8B8A8_UNORM',
}

# Per-stage binding map.  `role` is the name the shader uses; `offset` is the
# argument slot the capture recorded the binding at, taken from the stage source
# plus the capture's own payloads (see the stage graph finding).
#
# An optional fourth element overrides what a binding needs beyond its slot:
# for a buffer, the exact byte count the stage reads (a captured pointer may land
# inside a larger allocation); for a texture, the D3D12 view to declare when the
# shader writes exact bytes rather than relying on the UNORM conversion.
STAGES = {
    'downsample': {
        'shader': 'downsample_cs.cso',
        'srv': [('src', 'buffer', 24), ('colour', 'tex2d', 56)],
        'uav': [('out', 'tex2d', 32)],
    },
    # dec0: weights at arg 0, the token field at arg 24, then the four written
    # surfaces.  The native shader maps one thread per token, so its dispatch is
    # its own (like `downsample`'s per-pixel gather); the stage's output is what
    # the fixture compares.
    'dec0': {
        'shader': 'dec0_cs.cso',
        'threads': (64, 1, 1),
        'tokens': 480 * 272,
        # The reference's own arithmetic is approximate where it says so: the PTX
        # uses `ex2.approx`, `rcp.approx` and `div.approx`, each specified to a
        # couple of ULPs, and the surfaces quantise to 8 bits or f16.  So the
        # claim is agreement on at least 99.9% of samples; byte equality is *not*
        # claimed, and EXP-014 enumerates every residual, including the ~200
        # covariance samples per frame that sit on an e4m3 rounding boundary and
        # land on the other side of it.
        'expect': {'identical_min': 0.999},
        'srv': [('weights', 'buffer', 0, 8448),
                ('tokens', 'buffer', 24, 8355840)],
        'uav': [('gate', 'tex2d', 136, 'R32_UINT'),
                ('mix', 'tex2d', 144, 'R32_UINT'),
                ('feat', 'tex2d', 128, 'R32_UINT'),
                ('cov', 'tex2d', 152, 'R16G16B16A16_FLOAT')],
    },
    # enc0: the weight blob at arg 0, the colour texture at arg 80, and the token
    # field at arg 48 - which the kernel writes with `st.global`, so it is a
    # buffer, not a surface.  One thread per token (480 x 272), not the captured
    # 121 x 69 x 32: the captured grid is 8x8 pixels per block with two passes,
    # and the stage's output is what the fixture compares.
    #
    # The field is E4M3 bytes compared byte for byte.  The CPU oracle reproduces
    # all 8 355 840 of them (EXP-015), so the criterion is exactness, declared
    # here before the run.
    'enc0': {
        'shader': 'enc0_cs.cso',
        'threads': (64, 1, 1),
        'tokens': 480 * 272,
        'expect': {'identical_min': 1.0},
        'srv': [('weights', 'buffer', 0, 12416),
                ('colour', 'tex2d', 80)],
        'uav': [('tokens', 'buffer', 48, 8355840)],
    },
    # post_3_1: five textures in, three outputs - the 2880x1620 RGBA16F
    # reconstruction, the 2880x1620 packed 11-11-10 word buffer, and the
    # 1440x810 half-resolution surface.  The reconstruction surface is written
    # through an R32_UINT view (two words per pixel) because the captured f16
    # conversion truncates: the shader reproduces the bytes, not the conversion.
    #
    # The comparison floor is declared before the native run and is a floor, not
    # the claim: the CPU oracle misses one code at the 11-bit/f16 boundary for
    # ~1e-5 of samples because the reference's own ex2.approx/rcp.approx move a
    # value that sits exactly on a boundary.  The GPU's own approximations decide
    # those identically, so the stage then reaches exactness (EXP-016).
    'post_3_1': {
        'shader': 'post_cs.cso',
        'expect': {'identical_min': 0.9999},
        'srv': [('colour', 'tex2d', 0), ('gate', 'tex2d', 32),
                ('mix', 'tex2d', 40), ('cov', 'tex2d', 48),
                ('motion', 'tex2d', 176)],
        'uav': [('packed', 'buffer', 104, 18662400),
                ('history', 'tex2d', 96, 'R32_UINT', (5760, 1620)),
                ('half', 'tex2d', 112)],
    },
}


def fail(message):
    raise SystemExit(f'error: {message}')


# ---------------------------------------------------------------------------
# capture access


def launch(capture, stage, frame=1):
    """(directory, manifest fields) of one stage launch in one frame."""
    dirs = iv.stage_dirs(capture, stages=(stage,))
    entries = dirs.get(stage)
    if not entries:
        fail(f'the capture has no {stage} launch')
    if frame > len(entries):
        fail(f'the capture has {len(entries)} {stage} launches, wanted frame {frame}')
    directory = entries[frame - 1]
    lines = open(os.path.join(directory, 'manifest.txt')).read().split('\n')
    head = lines[1].split()
    return directory, {
        'kernel': lines[0].split()[1],
        'groups': [int(v) for v in head[1:4]],
        'threads': [int(v) for v in head[4:7]],
        'args_bytes': int(lines[2].split()[1]),
        'args': open(os.path.join(directory, 'args.bin'), 'rb').read(),
    }


def pointers(directory):
    """argument offset -> (allocation id, offset into it) for captured pointers."""
    table = {}
    for line in open(os.path.join(directory, 'manifest.txt')):
        q = line.split()
        if q and q[0] == 'pointer':
            table[int(q[1])] = (int(q[2]), int(q[3]))
    return table


def allocations(directory):
    return audit.allocations(directory)


def arg_pointer(args, offset):
    if offset + 8 > len(args):
        fail(f'argument offset {offset} is outside the {len(args)}-byte block')
    return struct.unpack_from('<Q', args, offset)[0]


def images_at(directory):
    """argument offset -> (image id, image record) for every image binding."""
    table, _ = audit.images(directory)
    return {binding['offset']: (image_id, image)
            for image_id, image in table.items() for binding in image['bindings']}


def image_record(directory, image_id, post=False):
    table, _ = audit.images(directory)
    if image_id not in table:
        fail(f'{directory} has no image {image_id}')
    image = dict(table[image_id])
    path = os.path.join(directory, 'gpu-output' if post else '', f'image-{image_id}.bin')
    if not os.path.exists(path):
        fail(f'missing image payload {path}')
    data = open(path, 'rb').read()
    if len(data) != image['bytes']:
        fail(f'{path} is {len(data)} bytes, the manifest declares {image["bytes"]}')
    image['data'] = data
    image['path'] = path
    return image


def image_format(image):
    """The D3D12 staging format for one captured image, with a stride check."""
    key = (image['format'], image['channels'])
    if key not in IMAGE_FORMATS:
        fail(f'no D3D12 staging format for backing 0x{image["format"]:02x} x{image["channels"]}')
    element = 2 if image['format'] == 0x10 else 4 if image['format'] == 0x20 else 1
    expected_row = image['width'] * image['channels'] * element
    if image['row_bytes'] != expected_row:
        fail(f'row stride {image["row_bytes"]} does not match {image["width"]}x'
             f'{image["channels"]} of 0x{image["format"]:02x} ({expected_row})')
    if image['depth'] != 1:
        fail(f'{image["width"]}x{image["height"]}x{image["depth"]}: this runner stages '
             '2D images only')
    return IMAGE_FORMATS[key]


# ---------------------------------------------------------------------------
# stage parameters


def downsample_params(args, fields):
    """down_m.comp's parameter block, from the slots d4r's own harness reads.

    d4r's `test_down_m.py` rebuilds the same struct with
    `struct.pack('<8i4ff', i2(0), i2(40), i2(64), i2(72), f2(80), f2(88), expo)`,
    which is what fixes these offsets.  The exposure is not in the block: d4r
    takes it from the 1x1 f16 texture at argument 48 (`res[48]`), and only the
    HDR variant reads it.
    """
    i2 = lambda o: struct.unpack_from('<2i', args, o)
    f2 = lambda o: struct.unpack_from('<2f', args, o)
    values = {
        'srcSize': list(i2(0)), 'outOrigin': list(i2(40)),
        'colOrigin': list(i2(64)), 'colSize': list(i2(72)),
        'out2rnd': list(f2(80)), 'rndOffs': list(f2(88)),
    }
    blob = (struct.pack('<4i', *values['srcSize'], *values['outOrigin'])
            + struct.pack('<4i', *values['colOrigin'], *values['colSize'])
            + struct.pack('<4f', *values['out2rnd'], *values['rndOffs'])
            + struct.pack('<4f', 0.0, 0.0, 0.0, 0.0))
    return blob, values


PARAMS = {'downsample': downsample_params}


def dec0_params(args, fields):
    """dec0's parameter block, from the slots its own PTX reads.

    The PTX loads `param+16` as the render size, `param+120` as the reset flag,
    and `param+160`/`param+164` as two f32s that the epilogue uses as sigma and
    rho.  `param+16` is (960, 544): the *padded* render height, not 540 - which is
    what fixes the token plane stride at 272 rows (EXP-014).
    """
    i2 = lambda o: struct.unpack_from('<2i', args, o)
    f1 = lambda o: struct.unpack_from('<f', args, o)[0]
    values = {'size': list(i2(16)), 'reset': args[120],
              'sigma': f1(160), 'rho': f1(164)}
    blob = (struct.pack('<2i', *values['size']) + struct.pack('<2i', 0, 0)
            + struct.pack('<2f', values['sigma'], values['rho'])
            + struct.pack('<2i', values['reset'], 0))
    return blob, values


PARAMS['dec0'] = dec0_params


def enc0_params(args, fields):
    """enc0's parameter block, from the slots its own PTX reads.

    `param+16` is the *padded* size (960, 544) that the store uses for its row
    and plane strides, `param+56` is the visible size (960, 540) the mirror
    folds against, and `param+312`/`+316` are the dither table's frame index and
    the reset flag.  Both sizes are needed because they differ - using the
    visible height for the plane shifted dec0's channels 32..63 by two rows
    (EXP-013).
    """
    i2 = lambda o: list(struct.unpack_from('<2i', args, o))
    values = {'padded': i2(16), 'visible': i2(56),
              'frame': struct.unpack_from('<I', args, 312)[0],
              'reset': args[316]}
    blob = (struct.pack('<2i', *values['visible'])
            + struct.pack('<2i', *values['padded'])
            + struct.pack('<2I', values['frame'], values['reset']))
    return blob, values


PARAMS['enc0'] = enc0_params


def post_params(args, fields):
    """post's cbuffer, from the slots its own PTX loads.

    The two sizes differ deliberately: `param+140` (2880x1620) is the
    reconstruction grid the position maps onto and the packed buffer is sized
    by, `param+148` (960x540) is the render grid the tile and the colour
    fallback clamp against.  `param+192` is the motion texture's declared size.
    """
    f = lambda o: struct.unpack_from('<f', args, o)[0]
    i2 = lambda o: list(struct.unpack_from('<2i', args, o))
    values = {'visible': i2(148), 'recon': i2(140), 'display': i2(192),
              'colour_origin': i2(8),
              'scale': [f(124), f(128), f(132), f(136)],
              'gain': f(120), 'mv_scale': [f(200), f(204)],
              'mv_offset': [f(208), f(212)], 'reset': args[168]}
    blob = (struct.pack('<2i', *values['visible'])
            + struct.pack('<2i', *values['recon'])
            + struct.pack('<2i', *values['display'])
            + struct.pack('<2i', *values['colour_origin'])
            + struct.pack('<4f', *values['scale'])
            + struct.pack('<f', values['gain'])
            + struct.pack('<2f', *values['mv_scale'])
            + struct.pack('<2f', *values['mv_offset'])
            + struct.pack('<I', values['reset']))
    return blob, values


PARAMS['post_3_1'] = post_params


def validate_stage(stage, record):
    """Reject a fixture whose geometry contradicts the stage it claims to be.

    A capture can be internally consistent and still not be the case a shader
    was written for: another resolution, a sub-rectangle, another packing.  These
    are the checks the reconstruction depends on, so a violation is an error;
    whatever they cannot establish is recorded as an assumption in EXP-012.
    """
    problems = []
    b = record['bindings']
    if stage == 'downsample':
        p = record['params']
        sw, sh = p['srcSize']
        ow, oh, _ = b['out']['geometry']
        if (ow * 3, oh * 3) != (sw * 2, sh * 2):
            problems.append(f'output {ow}x{oh} is not 2/3 of the packed source {sw}x{sh}')
        if b['src']['bytes'] != 4 * sw * sh:
            problems.append(f'the packed source is {b["src"]["bytes"]} B, expected '
                            f'4*{sw}*{sh} for the 11-11-10 word')
        if p['colSize'] != b['colour']['geometry'][:2]:
            problems.append(f'colour window {p["colSize"]} != colour image '
                            f'{b["colour"]["geometry"][:2]}')
        if p['outOrigin'] != [0, 0] or p['colOrigin'] != [0, 0]:
            problems.append('non-zero output/colour origin: the native shader implements '
                            'the full-frame case only')
        groups, threads = record['groups'], record['threads']
        if groups[0] * threads[0] < ow or groups[1] * threads[1] < oh:
            problems.append(f'grid {groups} x block {threads} does not cover {ow}x{oh}')
        if p['out2rnd'] != [0.5, 0.5]:
            problems.append(f'out2rnd {p["out2rnd"]} is not the render/output scale')
    if stage == 'dec0':
        p = record['params']
        w, h = p['size']
        plane = (h // 2) * (w // 2) * 32
        if b['tokens']['bytes'] != 2 * plane:
            problems.append(f'the token field is {b["tokens"]["bytes"]} B, expected '
                            f'2*{plane} for 2 planes of {h // 2}x{w // 2} tokens')
        if b['weights']['bytes'] != 8448:
            problems.append(f'the weight blob is {b["weights"]["bytes"]} B, expected '
                            f'8192 (64x128 e4m3) + 256 (128 f16 bias)')
        if p['reset'] not in (0, 1):
            problems.append(f'reset flag {p["reset"]} is not a boolean')
        for role in ('feat', 'gate', 'mix', 'cov'):
            if role not in b:
                problems.append(f'missing output {role}')
                continue
            geometry = b[role]['geometry'][:2]
            # the surfaces are the *unpadded* render rectangle; the parameter
            # block's height is the padded one (544 for a 540-line frame)
            if geometry[0] != w or geometry[1] > h:
                problems.append(f'{role} is {geometry}, expected {w} wide and at most '
                                f'{h} tall')
        for role in ('feat', 'gate', 'mix'):
            if role in b and b[role]['format'] != 'R32_UINT':
                problems.append(f'{role} must be written as exact bytes through an '
                                f'R32_UINT view, not {b[role]["format"]}')
        groups, threads = record['groups'], record['threads']
        tokens = (w // 2) * (h // 2)
        if groups[0] * threads[0] < tokens:
            problems.append(f'grid {groups} x block {threads} does not cover {tokens} '
                            'tokens')
    if stage == 'enc0':
        p = record['params']
        pw, ph = p['padded']
        vw, vh = p['visible']
        plane = (ph // 2) * (pw // 2) * 32
        if b['weights']['bytes'] != 12416:
            problems.append(f'the weight blob is {b["weights"]["bytes"]} B, expected '
                            f'12416 = 4096 (dither) + 8192 (128x64 e4m3) + 128 (bias)')
        if b['tokens']['bytes'] != 2 * plane:
            problems.append(f'the token field is {b["tokens"]["bytes"]} B, expected '
                            f'2*{plane} for 2 planes of {ph // 2}x{pw // 2} tokens '
                            f'from the padded height')
        if b['colour']['geometry'][:2] != [vw, vh]:
            problems.append(f'the colour texture is {b["colour"]["geometry"][:2]}, '
                            f'not the visible size {[vw, vh]}')
        if p['reset'] != 1:
            problems.append(f'reset is {p["reset"]}: the native shader implements the '
                            'reset frame only, where the history, previous-feature and '
                            'exposure slots are provably zero')
        if p['frame'] > 7:
            problems.append(f'frame {p["frame"]} is outside the 8-frame dither table')
        groups, threads = record['groups'], record['threads']
        tokens = (ph // 2) * (pw // 2)
        if groups[0] * threads[0] < tokens:
            problems.append(f'grid {groups} x block {threads} does not cover {tokens} '
                            'tokens')
    if stage == 'post_3_1':
        p = record['params']
        rw, rh = p['recon']
        vw, vh = p['visible']
        if b['colour']['geometry'][:2] != [vw, vh]:
            problems.append(f'the colour texture is {b["colour"]["geometry"][:2]}, '
                            f'not the render size {[vw, vh]}')
        for role in ('gate', 'mix', 'cov'):
            if b[role]['geometry'][:2] != [vw, vh]:
                problems.append(f'{role} is {b[role]["geometry"][:2]}, not the render '
                                f'size {[vw, vh]}')
        if b['packed']['bytes'] != rw * rh * 4:
            problems.append(f'the packed buffer is {b["packed"]["bytes"]} B, expected '
                            f'{rw * rh * 4} for {rw}x{rh} words')
        if b['history']['bytes'] != rw * rh * 8:
            problems.append(f'the reconstruction surface is {b["history"]["bytes"]} B, '
                            f'expected {rw * rh * 8} for {rw}x{rh} RGBA16F')
        if b['half']['bytes'] != (rw // 2) * (rh // 2) * 16:
            problems.append(f'the half-resolution surface is {b["half"]["bytes"]} B, '
                            f'expected {(rw // 2) * (rh // 2) * 16}')
        if p['reset'] != 1:
            problems.append(f'reset is {p["reset"]}: the native shader implements the '
                            'reset frame only, where the history sample is the colour '
                            'texel and the Catmull-Rom path is unreachable')
        if abs(p['scale'][0] - p['scale'][1]) > 0.0:
            problems.append(f'the x and y scales differ ({p["scale"][0]}, '
                            f'{p["scale"][1]}): this stage assumes a square ratio')
        groups, threads = record['groups'], record['threads']
        if groups[0] * threads[0] < rw or groups[1] * threads[1] * 2 < rh:
            problems.append(f'grid {groups} x block {threads} does not cover {rw}x{rh}')
    return problems


# ---------------------------------------------------------------------------
# fixture export


def export(capture, stage, out, frame):
    spec = STAGES.get(stage)
    if spec is None:
        fail(f'no fixture map for stage {stage!r}')
    directory, fields = launch(capture, stage, frame)
    args = fields['args']
    params, values = PARAMS[stage](args, fields)
    allocs = allocations(directory)
    at = images_at(directory)

    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'params.bin'), 'wb').write(params)
    threads = list(spec.get('threads') or fields['threads'])
    groups = list(spec.get('groups') or fields['groups'])
    if spec.get('tokens'):                       # one thread per token
        groups = [(-(-spec['tokens'] // threads[0])), 1, 1]
    lines = ['version 2', f'shader {spec["shader"]}',
             'threads {} {} {}'.format(*threads),
             'groups {} {} {}'.format(*groups),
             f'params params.bin {len(params)}']
    record = {
        'stage': stage, 'kernel': fields['kernel'], 'frame': frame,
        'capture_directory': directory, 'params': values, 'bindings': {},
        'groups': groups, 'threads': threads,
        'captured_groups': fields['groups'], 'captured_threads': fields['threads'],
    }
    if spec.get('expect'):
        record['expect'] = spec['expect']

    def emit_input(role, entry):
        record['bindings'][role] = entry
        if entry['kind'] == 'buffer':
            lines.append(f'srv buffer {len(record["bindings"]) - 1} in-{role}.bin '
                         f'{entry["bytes"]}')
        else:
            lines.append(f'srv tex2d {len(record["bindings"]) - 1} {entry["format"]} '
                         f'{entry["geometry"][0]} {entry["geometry"][1]} in-{role}.bin')

    for entry in spec['srv']:
        role, kind, offset = entry[0], entry[1], entry[2]
        if kind == 'buffer':
            want = entry[3] if len(entry) > 3 else None
            pointer = arg_pointer(args, offset)
            match = [(i, a) for i, a in allocs.items()
                     if a['base'] <= pointer < a['base'] + a['size']]
            if not match:
                fail(f'{role}: argument {offset} points at 0x{pointer:x}, which is not a '
                     'captured allocation')
            index, alloc = match[0]
            skip = pointer - alloc['base']
            data = open(alloc['pre'], 'rb').read()[skip:]
            if want is not None:
                if len(data) < want:
                    fail(f'{role}: the allocation holds {len(data)} B from the pointer, '
                         f'the stage reads {want}')
                data = data[:want]
            open(os.path.join(out, f'in-{role}.bin'), 'wb').write(data)
            emit_input(role, {'kind': 'buffer', 'arg': offset, 'allocation': index,
                              'base': hex(alloc['base']), 'offset': skip,
                              'bytes': len(data)})
        else:
            if offset not in at:
                fail(f'{role}: no captured image at argument offset {offset}')
            image_id, _ = at[offset]
            image = image_record(directory, image_id)
            open(os.path.join(out, f'in-{role}.bin'), 'wb').write(image['data'])
            emit_input(role, {
                'kind': 'image', 'arg': offset, 'image': image_id,
                'format': image_format(image), 'backing_format': hex(image['format']),
                'geometry': [image['width'], image['height'], image['depth']],
                'row_bytes': image['row_bytes'], 'bytes': image['bytes']})

    for entry in spec['uav']:
        role, kind, offset = entry[0], entry[1], entry[2]
        override = entry[3] if len(entry) > 3 else None
        if kind == 'buffer':
            # a `st.global` output: the pointer resolves into a captured
            # allocation, and the reference is that allocation's post-launch
            # image from the same offset
            want = entry[3] if len(entry) > 3 else None
            pointer = arg_pointer(args, offset)
            match = [(i, a) for i, a in allocs.items()
                     if a['base'] <= pointer < a['base'] + a['size']]
            if not match:
                fail(f'{role}: argument {offset} points at 0x{pointer:x}, which is '
                     'not a captured allocation')
            index, alloc = match[0]
            if not alloc['post']:
                fail(f'{role}: allocation {index} has no post-launch image to compare '
                     'against')
            skip = pointer - alloc['base']
            before = open(alloc['pre'], 'rb').read()[skip:]
            after = open(alloc['post'], 'rb').read()[skip:]
            length = want if want is not None else len(after)
            if len(after) < length:
                fail(f'{role}: the allocation holds {len(after)} B from the pointer, '
                     f'the stage writes {length}')
            open(os.path.join(out, f'ref-{role}.bin'), 'wb').write(after[:length])
            record['bindings'][role] = {
                'kind': 'buffer', 'arg': offset, 'allocation': index,
                'base': hex(alloc['base']), 'offset': skip, 'output': True,
                'format': 'R8_UINT', 'geometry': [length, 1, 1],
                'bytes': length, 'reference': f'ref-{role}.bin',
                'changed_bytes': int(np.count_nonzero(
                    np.frombuffer(before[:length], np.uint8)
                    != np.frombuffer(after[:length], np.uint8)))}
            lines.append(f'uav buffer {len(record["bindings"]) - 1} out-{role}.bin '
                         f'{length}')
            continue
        if kind != 'tex2d' or offset not in at:
            fail(f'{role}: no captured output image at argument offset {offset}')
        image_id, _ = at[offset]
        image = image_record(directory, image_id)
        reference = image_record(directory, image_id, post=True)
        declared = override or image_format(image)
        geometry = entry[4] if len(entry) > 4 else None
        width = geometry[0] if geometry else image['width']
        height = geometry[1] if geometry else image['height']
        if declared != image_format(image) or geometry:
            # an override changes the typed view, never the bytes
            element = 4 if declared.startswith(('R32', 'R10')) else 2
            if width * height * element != image['bytes']:
                fail(f'{role}: the typed view {declared} {width}x{height} needs '
                     f'{width * height * element} B, the capture has {image["bytes"]}')
        open(os.path.join(out, f'ref-{role}.bin'), 'wb').write(reference['data'])
        record['bindings'][role] = {
            'kind': 'image', 'arg': offset, 'image': image_id, 'output': True,
            'format': declared, 'backing_format': hex(image['format']),
            'geometry': [width, height, 1],
            'row_bytes': image['row_bytes'], 'bytes': image['bytes'],
            'reference': f'ref-{role}.bin',
            'changed_bytes': int(np.count_nonzero(
                np.frombuffer(image['data'], np.uint8)
                != np.frombuffer(reference['data'], np.uint8)))}
        lines.append(f'uav tex2d {len(record["bindings"]) - 1} '
                     f'{record["bindings"][role]["format"]} '
                     f'{width} {height} out-{role}.bin')

    problems = validate_stage(stage, record)
    if problems:
        fail('the captured launch is not the case this stage implements:\n  '
             + '\n  '.join(problems))

    open(os.path.join(out, 'stage.txt'), 'w').write('\n'.join(lines) + '\n')
    record['manifest'] = 'stage.txt'
    with open(os.path.join(out, 'fixture.json'), 'w') as handle:
        json.dump(record, handle, indent=2)
    print(f'{stage}: {fields["kernel"]} frame {frame}, groups {fields["groups"]}, '
          f'threads {fields["threads"]}')
    for role, entry in record['bindings'].items():
        if entry['kind'] == 'buffer':
            print(f'  {role:8s} buffer {entry["bytes"]:>10d} B  arg {entry["arg"]}')
        else:
            print(f'  {role:8s} {entry["format"]:20s} {entry["geometry"][0]}x'
                  f'{entry["geometry"][1]}  arg {entry["arg"]}'
                  f'{"  (output)" if entry.get("output") else ""}')
    print(f'  -> {out}')
    return 0


# ---------------------------------------------------------------------------
# B4.1: transport round trip


ROUNDTRIP = [
    # (label, stage, frame, role, kind, argument slot, expected format, post)
    ('downsample src, packed 18.6 MB', 'downsample', 1, 'src', 'buffer', 24, None, False),
    ('downsample colour 960x540 f16x4', 'downsample', 1, 'colour', 'tex2d', 56,
     'R16G16B16A16_FLOAT', False),
    ('downsample frame 1920x1080 f16x4', 'downsample', 1, 'out', 'tex2d', 32,
     'R16G16B16A16_FLOAT', True),
    ('enc0 colour 960x540 f16x4', 'enc0', 1, 'colour', 'tex2d', 80,
     'R16G16B16A16_FLOAT', False),
    ('enc0 depth 960x540 f32', 'enc0', 1, 'depth', 'tex2d', 136, 'R32_FLOAT', False),
    ('enc0 motion 960x540 f16x2', 'enc0', 1, 'motion', 'tex2d', 128, 'R16G16_FLOAT', False),
    ('dec0 feature 960x540 snorm8x4', 'dec0', 1, 'feature', 'tex2d', 128,
     'R8G8B8A8_UNORM', True),
    ('post history 2880x1620 f16x4', 'post_3_1', 1, 'history', 'tex2d', 96,
     'R16G16B16A16_FLOAT', True),
]


def write_copy_fixture(directory, lines, data):
    os.makedirs(directory, exist_ok=True)
    open(os.path.join(directory, 'params.bin'), 'wb').write(b'\0' * 16)
    open(os.path.join(directory, 'in.bin'), 'wb').write(data)
    open(os.path.join(directory, 'stage.txt'), 'w').write('\n'.join(lines) + '\n')


def run_probe(probe, shaders, fixture, results, repeats=1):
    """Execute one fixture `repeats` times in separate processes."""
    outputs = []
    for index in range(repeats):
        directory = results if index == 0 else f'{results}-repeat{index}'
        command = [probe, '--fixture', os.path.join(fixture, 'stage.txt'), '--out', directory]
        if shaders:
            command += ['--shaders', shaders]
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode:
            raise SystemExit(f'error: {probe} failed on {os.path.basename(fixture)}:\n'
                             f'{done.stdout[-400:]}{done.stderr[-400:]}')
        outputs.append(directory)
    return outputs


def roundtrip(capture, out, probe, shaders):
    os.makedirs(out, exist_ok=True)
    rows = []
    for label, stage, frame, role, kind, offset, want, post in ROUNDTRIP:
        directory, fields = launch(capture, stage, frame)
        if kind == 'buffer':
            pointer = arg_pointer(fields['args'], offset)
            match = [(i, a) for i, a in allocations(directory).items() if a['base'] == pointer]
            if not match:
                fail(f'{label}: argument {offset} is not a captured allocation')
            data = open(match[0][1]['pre'], 'rb').read()
            lines = ['version 2', 'shader image_copy_buf.cso', 'threads 64 1 1',
                     f'groups {(len(data) // 4 + 63) // 64} 1 1', 'params params.bin 16',
                     f'srv buffer 0 in.bin {len(data)}',
                     f'uav buffer 0 out.bin {len(data)}']
        else:
            at = images_at(directory)
            if offset not in at:
                fail(f'{label}: no captured image at the mapped slot')
            image_id, _ = at[offset]
            image = image_record(directory, image_id, post=post)
            fmt = image_format(image)
            if want and fmt != want:
                fail(f'{label}: captured format resolves to {fmt}, expected {want}')
            data = image['data']
            lines = ['version 2', 'shader image_copy_tex.cso', 'threads 16 16 1',
                     f'groups {(image["width"] + 15) // 16} '
                     f'{(image["height"] + 15) // 16} 1', 'params params.bin 16',
                     f'srv tex2d 0 {fmt} {image["width"]} {image["height"]} in.bin',
                     f'uav tex2d 0 {fmt} {image["width"]} {image["height"]} out.bin']
        fixture = os.path.join(out, f'copy-{role}-{stage}')
        write_copy_fixture(fixture, lines, data)
        results = os.path.join(fixture, 'results')
        try:
            run_probe(probe, shaders, fixture, results)
        except SystemExit as error:
            rows.append({'resource': label, 'pass': False, 'error': str(error)})
            print(f'FAIL  {label}: {str(error).strip()[-200:]}')
            continue
        written = open(os.path.join(results, 'out.bin'), 'rb').read()
        differing = (-1 if len(written) != len(data)
                     else int(np.count_nonzero(np.frombuffer(written, np.uint8)
                                               != np.frombuffer(data, np.uint8))))
        ok = len(written) == len(data) and differing == 0
        rows.append({'resource': label, 'bytes': len(data), 'written': len(written),
                     'differing': differing, 'pass': ok})
        print(f'{"PASS" if ok else "FAIL"}  {label:34s} {len(data):>9d} B  '
              f'{differing} byte(s) differ')
    problems = [row for row in rows if not row['pass']]
    json.dump({'roundtrip': rows, 'pass': not problems},
              open(os.path.join(out, 'roundtrip.json'), 'w'), indent=2)
    print(f'\n{len(rows) - len(problems)}/{len(rows)} resources round-tripped unchanged')
    return 0 if not problems else 1


# ---------------------------------------------------------------------------
# B4.7: differential comparison


# bytes per pixel, for the spatial localisation of differences
PIXEL_BYTES = {'R16G16B16A16_FLOAT': 8, 'R8G8B8A8_UNORM': 4, 'R32G32B32A32_FLOAT': 16,
               'R8G8B8A8_UINT': 4, 'R32_UINT': 4, 'R16G16_FLOAT': 4,
               'R32G32_FLOAT': 8, 'R16_FLOAT': 2, 'R32_FLOAT': 4}


def compare(native, reference, role, fmt):
    """Exact-byte and numerical comparison of one output image."""
    row = {'role': role, 'format': fmt, 'bytes': len(reference)}
    if len(native) != len(reference):
        row['pass'] = False
        row['error'] = f'size mismatch: {len(native)} vs {len(reference)}'
        return row
    a = np.frombuffer(native, np.uint8)
    b = np.frombuffer(reference, np.uint8)
    differing = int(np.count_nonzero(a != b))
    row.update({'differing_bytes': differing, 'identical_bytes': len(b) - differing,
                'pass': differing == 0, 'exact': differing == 0})
    if fmt in ('R16G16B16A16_FLOAT', 'R16G16_FLOAT', 'R16_FLOAT'):
        x = np.frombuffer(native, np.float16).astype(np.float64)
        y = np.frombuffer(reference, np.float16).astype(np.float64)
        finite = np.isfinite(x)
        delta = np.abs(x - y)[finite]
        peak = float(np.abs(y).max()) or 1.0
        steps = np.abs(np.frombuffer(native, np.uint16).astype(np.int64)
                       - np.frombuffer(reference, np.uint16).astype(np.int64))
        # +0 and -0 are numerically equal but 32768 codes apart in the raw bits,
        # so a bit-pattern ULP count would report them as the largest possible
        # difference.  The stage writes a signed product, so the sign of a zero
        # is not a claim about the value.
        zero = (np.frombuffer(native, np.float16) == 0) & \
               (np.frombuffer(reference, np.float16) == 0)
        steps = np.where(zero, 0, steps)
        row.update({
            'nonfinite': int(x.size - finite.sum()),
            'identical_f16': float(np.mean(x == y)),
            'max_abs': float(delta.max()) if delta.size else 0.0,
            'rms': float(np.sqrt((delta ** 2).mean())) if delta.size else 0.0,
            'rel_rms': float(np.sqrt((delta ** 2).mean()) / peak) if delta.size else 0.0,
            'max_rel_to_peak': float(delta.max() / peak) if delta.size else 0.0,
            'ulp_histogram': {str(int(k)): int(v) for k, v in
                              zip(*np.unique(steps, return_counts=True))},
            'max_step': int(steps.max()),
            'within_one_step': float(np.mean(steps <= 1)),
        })
    elif fmt in ('R8G8B8A8_UNORM', 'R8G8B8A8_UINT', 'R32_UINT'):
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
        row['max_abs_code'] = int(delta.max())
    return row


def localise(native, reference, width, height, pixel_bytes):
    """Where the differing pixels are, so an error can be attributed."""
    a = np.frombuffer(native, np.uint8).reshape(height, width, pixel_bytes)
    b = np.frombuffer(reference, np.uint8).reshape(height, width, pixel_bytes)
    bad = np.any(a != b, axis=2)
    rows, cols = np.nonzero(bad)
    return {
        'pixels': int(bad.size), 'differing_pixels': int(bad.sum()),
        'first_row': int(rows.min()) if rows.size else None,
        'last_row': int(rows.max()) if rows.size else None,
        'first_col': int(cols.min()) if cols.size else None,
        'last_col': int(cols.max()) if cols.size else None,
        'rows_touched': int(np.unique(rows).size) if rows.size else 0,
        'cols_touched': int(np.unique(cols).size) if cols.size else 0,
        'border_only': bool(rows.size and (
            (rows < 2).all() or (rows >= height - 2).all()
            or (cols < 2).all() or (cols >= width - 2).all())),
    }


def check(fixture, results, repeat=None, json_out=None):
    record = json.load(open(os.path.join(fixture, 'fixture.json')))
    rows, problems = [], []
    for role, entry in record['bindings'].items():
        if not entry.get('output'):
            continue
        native_path = os.path.join(results, f'out-{role}.bin')
        if not os.path.exists(native_path):
            problems.append(f'{role}: the native run wrote no {os.path.basename(native_path)}')
            continue
        native = open(native_path, 'rb').read()
        reference = open(os.path.join(fixture, f'ref-{role}.bin'), 'rb').read()
        row = compare(native, reference, role, entry['format'])
        width, height = entry['geometry'][:2]
        if row.get('differing_bytes') and len(native) == len(reference):
            row['localisation'] = localise(native, reference, width, height,
                                           PIXEL_BYTES.get(entry['format'], 1))
        if repeat:
            again = os.path.join(repeat, f'out-{role}.bin')
            if os.path.exists(again):
                row['repeat_identical'] = open(again, 'rb').read() == native
                if not row['repeat_identical']:
                    problems.append(f'{role}: two runs of the same fixture differ')
        rows.append(row)
        # The acceptance criterion is declared by the fixture, before the run.
        expect = record.get('expect') or {}
        failures = []
        if not row.get('pass'):
            identical = 1 - row['differing_bytes'] / max(row['bytes'], 1)
            if 'identical_f16' in row:
                identical = row['identical_f16']
            if identical < expect.get('identical_min', 1.0):
                failures.append(f'{100 * identical:.4f}% identical, expected at least '
                                f'{100 * expect["identical_min"]:.4f}%')
            if expect.get('max_ulp') is not None and 'max_step' in row:
                if row['max_step'] > expect['max_ulp']:
                    failures.append(f'max {row["max_step"]} ULP, expected at most '
                                    f'{expect["max_ulp"]}')
            if not failures:
                row['within_declared_bound'] = True
        for failure in failures:
            problems.append(f'{row["role"]}: {failure}')
        if failures:
            row['declared_bound_failed'] = failures

    for row in rows:
        identical = 100 * (1 - row['differing_bytes'] / row['bytes'])
        if 'identical_f16' in row:
            print(f"{row['role']:6s} {row['format']:20s} bytes identical "
                  f"{identical:8.4f}%  f16 identical {100 * row['identical_f16']:8.4f}%  "
                  f"max|d| {row['max_abs']:.6g}  rms {row['rms']:.3g}  "
                  f"rel_rms {row['rel_rms']:.3g}  <=1 step "
                  f"{100 * row['within_one_step']:.4f}%  nonfinite {row['nonfinite']}")
            print(f"       ulp histogram {row['ulp_histogram']}")
        else:
            print(f"{row['role']:6s} {row['format']:20s} bytes identical {identical:8.4f}%  "
                  f"max code delta {row.get('max_abs_code')}")
        if row.get('localisation'):
            loc = row['localisation']
            print(f"       differing pixels {loc['differing_pixels']}/{loc['pixels']}  rows "
                  f"{loc['first_row']}..{loc['last_row']} ({loc['rows_touched']})  cols "
                  f"{loc['first_col']}..{loc['last_col']} ({loc['cols_touched']})  "
                  f"border_only={loc['border_only']}")
        if 'repeat_identical' in row:
            print(f"       repeat run identical: {row['repeat_identical']}")
    print(f'\n{len(problems)} problem(s)')
    for problem in problems:
        print(f'  {problem}')
    result = {'fixture': fixture, 'stage': record['stage'], 'kernel': record['kernel'],
              'results': rows, 'problems': problems, 'pass': bool(rows) and not problems}
    if json_out:
        json.dump(result, json_out, indent=2)
    return 0 if result['pass'] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    r = sub.add_parser('roundtrip')
    r.add_argument('capture')
    r.add_argument('--out', required=True)
    r.add_argument('--probe',
                   default=os.path.join('build', 'native', 'Release', 'm_image_stage.exe'))
    r.add_argument('--shaders', default=os.path.join(REPO, 'build', 'native'))
    e = sub.add_parser('export')
    e.add_argument('capture')
    e.add_argument('--stage', required=True)
    e.add_argument('--out', required=True)
    e.add_argument('--frame', type=int, default=1)
    c = sub.add_parser('check')
    c.add_argument('fixture')
    c.add_argument('--results', default=None)
    c.add_argument('--repeat', default=None)
    c.add_argument('--json', type=argparse.FileType('w'), default=None)
    u = sub.add_parser('run')
    u.add_argument('fixture')
    u.add_argument('--probe',
                   default=os.path.join('build', 'native', 'Release', 'm_image_stage.exe'))
    u.add_argument('--shaders', default=os.path.join(REPO, 'build', 'native'))
    u.add_argument('--repeats', type=int, default=2,
                   help='separate processes; the second is the determinism check')
    u.add_argument('--json', type=argparse.FileType('w'), default=None)
    a = ap.parse_args()
    if a.command == 'roundtrip':
        return roundtrip(a.capture, a.out, a.probe, a.shaders)
    if a.command == 'export':
        return export(a.capture, a.stage, a.out, a.frame)
    if a.command == 'run':
        outputs = run_probe(a.probe, a.shaders, a.fixture,
                            os.path.join(a.fixture, 'results'), a.repeats)
        return check(a.fixture, outputs[0],
                     outputs[1] if len(outputs) > 1 else None, a.json)
    return check(a.fixture, a.results or os.path.join(a.fixture, 'results'), a.repeat, a.json)


if __name__ == '__main__':
    sys.exit(main())
