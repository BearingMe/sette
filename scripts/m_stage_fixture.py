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
STAGES = {
    'downsample': {
        'shader': 'downsample_cs.cso',
        'srv': [('src', 'buffer', 24), ('colour', 'tex2d', 56)],
        'uav': [('out', 'tex2d', 32)],
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
    lines = ['version 2', f'shader {spec["shader"]}',
             'threads {} {} {}'.format(*fields['threads']),
             'groups {} {} {}'.format(*fields['groups']),
             f'params params.bin {len(params)}']
    record = {
        'stage': stage, 'kernel': fields['kernel'], 'frame': frame,
        'capture_directory': directory, 'params': values, 'bindings': {},
        'groups': fields['groups'], 'threads': fields['threads'],
    }

    def emit_input(role, entry):
        record['bindings'][role] = entry
        if entry['kind'] == 'buffer':
            lines.append(f'srv buffer {len(record["bindings"]) - 1} in-{role}.bin '
                         f'{entry["bytes"]}')
        else:
            lines.append(f'srv tex2d {len(record["bindings"]) - 1} {entry["format"]} '
                         f'{entry["geometry"][0]} {entry["geometry"][1]} in-{role}.bin')

    for role, kind, offset in spec['srv']:
        if kind == 'buffer':
            pointer = arg_pointer(args, offset)
            match = [(i, a) for i, a in allocs.items() if a['base'] == pointer]
            if not match:
                fail(f'{role}: argument {offset} points at 0x{pointer:x}, which is not a '
                     'captured allocation')
            index, alloc = match[0]
            data = open(alloc['pre'], 'rb').read()
            open(os.path.join(out, f'in-{role}.bin'), 'wb').write(data)
            emit_input(role, {'kind': 'buffer', 'arg': offset, 'allocation': index,
                              'base': hex(alloc['base']), 'bytes': len(data)})
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

    for role, kind, offset in spec['uav']:
        if kind != 'tex2d' or offset not in at:
            fail(f'{role}: no captured output image at argument offset {offset}')
        image_id, _ = at[offset]
        image = image_record(directory, image_id)
        reference = image_record(directory, image_id, post=True)
        open(os.path.join(out, f'ref-{role}.bin'), 'wb').write(reference['data'])
        record['bindings'][role] = {
            'kind': 'image', 'arg': offset, 'image': image_id, 'output': True,
            'format': image_format(image), 'backing_format': hex(image['format']),
            'geometry': [image['width'], image['height'], image['depth']],
            'row_bytes': image['row_bytes'], 'bytes': image['bytes'],
            'reference': f'ref-{role}.bin',
            'changed_bytes': int(np.count_nonzero(
                np.frombuffer(image['data'], np.uint8)
                != np.frombuffer(reference['data'], np.uint8)))}
        lines.append(f'uav tex2d {len(record["bindings"]) - 1} '
                     f'{record["bindings"][role]["format"]} '
                     f'{image["width"]} {image["height"]} out-{role}.bin')

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
               'R16G16_FLOAT': 4, 'R32G32_FLOAT': 8, 'R16_FLOAT': 2, 'R32_FLOAT': 4}


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
    elif fmt == 'R8G8B8A8_UNORM':
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
        if not row['pass']:
            problems.append(f'{role}: {row.get("differing_bytes")} of {len(reference)} '
                            'bytes differ')

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
