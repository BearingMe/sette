#!/usr/bin/env python3
"""M05 (B1/B2/B3): validate the image-backed stages against independent evidence.

The capture now records the CUDA arrays behind every texture and surface a
launch binds.  This script checks the parts of that evidence that have an
independent oracle - the probe's own synthetic inputs and its D3D12 readback -
so a "PASS" here is not the capture agreeing with itself:

  * `enc0`'s colour texture must equal the probe's checkerboard pattern over
    every pixel and all four channels (`tools/windows/d3d12_dlss_probe.cpp`);
  * `enc0`'s depth must be the constant 0.5 it uploads, motion must be zero,
    and on the first frame the recurrent feature and half-resolution history
    must be zero (they start empty);
  * `dec0` must write its four surfaces with the geometries and formats d4r's
    `dec0_m.comp` declares, and each must actually change;
  * `post` must write its two history surfaces and its packed buffer;
  * `downsample`'s output array must be byte-identical to the frame the probe
    read back from the D3D12 output texture - a different code path entirely;
  * the in-frame image edges `dec0 -> post` and the temporal edges
    `frame N -> frame N+1` must be byte-identical.

Roles are decided by content, so a stage that rewrites a surface with identical
bytes is indistinguishable from one that does not touch it; that limitation is
reported rather than hidden.

usage: m_image_validate.py CAPTURE --outputs DIR [--frames N] [--json OUT]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import m_capture_audit as audit  # noqa: E402
import m_weights as mw          # noqa: E402

FRAME_BYTES = 1920 * 1080 * 8
# d4r dec0_m.comp bindings 3-6: featOut rgba8_snorm, gateOut rgba8, mixOut
# rgba8, covOut rgba16f.  Formats are the driver's backing storage codes.
DEC0_SURFACES = [
    {'label': 'feat', 'format': 0x08, 'channels': 4, 'original': 200, 'row': 3840},
    {'label': 'gate', 'format': 0x01, 'channels': 4, 'original': 194, 'row': 3840},
    {'label': 'mix', 'format': 0x01, 'channels': 4, 'original': 194, 'row': 3840},
    {'label': 'cov', 'format': 0x10, 'channels': 4, 'original': 0, 'row': 7680},
]
TUBE = 6


def stage_dirs(capture, stages=('enc0', 'dec0', 'post_3_1', 'downsample')):
    """stage -> list of replay directories in frame order."""
    out = {}
    for name in os.listdir(capture):
        directory = os.path.join(capture, name)
        if not os.path.isdir(directory) or not name.startswith('replay-'):
            continue
        kernel = open(os.path.join(directory, 'manifest.txt')).readline().split()[1]
        variant = mw.parse_kernel(kernel)
        if variant is None or variant['stage'] not in stages:
            continue
        sequence = int(name.split('-')[1])
        out.setdefault(variant['stage'], []).append((sequence, directory))
    for name in out:
        out[name].sort()
    return {name: [d for _, d in entries] for name, entries in out.items()}


def read_image(directory, image_id, post=False):
    path = os.path.join(directory, 'gpu-output' if post else '',
                        f'image-{image_id}.bin')
    return np.fromfile(path, np.uint8) if os.path.exists(path) else None


def images_of(directory):
    table, _ = audit.images(directory)
    return table


def changed(directory, image_id, table):
    pre = read_image(directory, image_id)
    post = read_image(directory, image_id, post=True)
    if pre is None or post is None or pre.size != post.size:
        return None
    return int(np.count_nonzero(pre != post))


def by_arg(table):
    """argument offset -> (image id, image record)."""
    return {binding['offset']: (image_id, image)
            for image_id, image in table.items() for binding in image['bindings']}


def check_enc0(directory, problems, rows):
    table = images_of(directory)
    at = by_arg(table)
    colour = read_image(directory, at[80][0])
    if colour is None:
        problems.append('enc0: colour texture not captured')
        return
    planes = colour.view(np.uint16).reshape(540, 960, 4)
    y = np.arange(540)[:, None]
    x = np.arange(960)[None, :]
    want = [
        np.broadcast_to(np.where(((x // 16 + y // 16) & 1) != 0, 0x3A00, 0x3400),
                        (540, 960)),
        np.broadcast_to(np.where((x & 32) != 0, 0x3800, 0x3400), (540, 960)),
        np.broadcast_to(np.where((y & 32) != 0, 0x3A00, 0x3800), (540, 960)),
        np.full((540, 960), 0x3C00),
    ]
    differing = [int(np.count_nonzero(planes[:, :, c] != want[c])) for c in range(4)]
    rows.append({'check': 'enc0 colour == probe pattern', 'differing': differing,
                 'pixels': 540 * 960, 'pass': not any(differing)})
    if any(differing):
        problems.append(f'enc0 colour differs from the probe pattern: {differing}')

    depth = read_image(directory, at[136][0]).view(np.float32)
    ok = bool(np.all(depth == np.float32(0.5)))
    rows.append({'check': 'enc0 depth == 0.5 everywhere', 'pass': ok})
    if not ok:
        problems.append('enc0 depth is not the constant 0.5 the probe uploads')

    motion = read_image(directory, at[128][0]).view(np.float16)
    ok = bool(np.all(motion == 0))
    rows.append({'check': 'enc0 motion == 0', 'pass': ok})
    if not ok:
        problems.append('enc0 motion is not zero')

    # Frame 1 only: the recurrent feature and half-resolution history start empty.
    feature = read_image(directory, at[160][0]).view(np.int8)
    history = read_image(directory, at[152][0]).view(np.float32)
    ok = bool(np.all(feature == 0) and np.all(history == 0))
    rows.append({'check': 'enc0 first-frame feature and history start at zero',
                 'feature_nonzero': int(np.count_nonzero(feature)),
                 'history_nonzero': int(np.count_nonzero(history)), 'pass': ok})
    if not ok:
        problems.append('enc0 frame-1 feature/history are not empty')

    surfaces = sorted(offset for offset in at if offset >= 280)
    rows.append({'check': 'enc0 binds three motion surfaces',
                 'offsets': surfaces, 'pass': len(surfaces) == 3})
    if len(surfaces) != 3:
        problems.append(f'enc0 motion surfaces: expected 3, found {len(surfaces)}')
    else:
        unchanged = [offset for offset in surfaces
                     if changed(directory, at[offset][0], table) == 0]
        rows.append({'check': 'enc0 motion surfaces unchanged by this dispatch',
                     'unchanged_offsets': unchanged,
                     'note': 'a rewrite with identical bytes is indistinguishable'})
    return table


def check_dec0(directory, problems, rows):
    table = images_of(directory)
    surfaces = [(image_id, image) for image_id, image in sorted(table.items())
                if image['kind'] == 'surf']
    if len(surfaces) != 4:
        problems.append(f'dec0: expected 4 surfaces, found {len(surfaces)}')
        return table
    for (image_id, image), want in zip(surfaces, DEC0_SURFACES):
        ok = (image['width'] == 960 and image['height'] == 540
              and image['format'] == want['format']
              and image['channels'] == want['channels']
              and image['original'] == want['original']
              and image['row_bytes'] == want['row'])
        delta = changed(directory, image_id, table)
        rows.append({'check': f"dec0 {want['label']} surface", 'format': image['format'],
                     'original': image['original'], 'row_bytes': image['row_bytes'],
                     'changed_bytes': delta, 'pass': bool(ok and delta)})
        if not ok:
            problems.append(f"dec0 {want['label']}: geometry/format {image} does not match "
                            f"dec0_m.comp ({want})")
        if not delta:
            problems.append(f"dec0 {want['label']}: no bytes changed by the dispatch")
    return table


def check_post(directory, problems, rows):
    table = images_of(directory)
    surfaces = [(image_id, image) for image_id, image in sorted(table.items())
                if image['kind'] == 'surf']
    if len(surfaces) != 2:
        problems.append(f'post: expected 2 output surfaces, found {len(surfaces)}')
        return table
    (hist_id, hist), (half_id, half) = surfaces
    delta_hist = changed(directory, hist_id, table)
    delta_half = changed(directory, half_id, table)
    ok = (hist['width'], hist['height'], hist['format']) == (2880, 1620, 0x10) \
        and (half['width'], half['height'], half['original']) == (1440, 810, 80)
    rows.append({'check': 'post history surface 2880x1620 half4', 'changed_bytes': delta_hist,
                 'pass': bool(ok and delta_hist)})
    rows.append({'check': 'post half surface 1440x810 packed 10:10:10:2',
                 'changed_bytes': delta_half, 'pass': bool(ok and delta_half)})
    if not ok:
        problems.append(f'post surfaces do not match post_m.comp: {hist} / {half}')
    if not delta_hist or not delta_half:
        problems.append('post did not write both history surfaces')
    return table


def check_downsample(directory, output_dir, frame, problems, rows):
    table = images_of(directory)
    surfaces = [(image_id, image) for image_id, image in sorted(table.items())
                if image['kind'] == 'surf']
    if not surfaces:
        problems.append('downsample: no output surface captured')
        return
    image_id, image = surfaces[0]
    post = read_image(directory, image_id, post=True)
    path = os.path.join(output_dir, f'output-{frame - 1}.rgba16f')
    if post is None:
        problems.append('downsample: output surface not captured')
        return
    if not os.path.exists(path) or os.path.getsize(path) != FRAME_BYTES:
        rows.append({'check': f'downsample output == output-{frame - 1}.rgba16f',
                     'pass': None, 'note': 'probe readback missing or incomplete'})
        return
    reference = np.fromfile(path, np.uint8)
    differing = int(np.count_nonzero(post != reference))
    rows.append({'check': f'downsample output == output-{frame - 1}.rgba16f',
                 'bytes': int(post.size), 'differing': differing,
                 'format': image['format'], 'pass': differing == 0})
    if differing:
        problems.append(f'downsample output differs from the probe readback: '
                        f'{differing} of {post.size} bytes')


def temporal_edges(capture, dirs, frames, problems, rows):
    """Frame N's written surfaces must be frame N+1's bound inputs, byte for byte."""
    def frame(stage, index):
        entries = dirs.get(stage, [])
        return entries[index] if index < len(entries) else None

    pairs = [
        ('dec0 feature -> enc0 recurrent feature', 'dec0', 2, 'enc0', 160, 7),
        ('post history -> post history input', 'post_3_1', 10, 'post_3_1', None, 6),
        ('post half-res -> enc0 half-res input', 'post_3_1', 11, 'enc0', 152, 6),
        ('enc0 motion A -> enc0 motion input', 'enc0', 8, 'enc0', 128, 4),
    ]
    for label, producer_stage, producer_image, consumer_stage, arg, consumer_image in pairs:
        ok_frames = 0
        for index in range(frames - 1):
            producer = frame(producer_stage, index)
            consumer = frame(consumer_stage, index + 1)
            if producer is None or consumer is None:
                continue
            a = read_image(producer, producer_image, post=True)
            b = read_image(consumer, consumer_image)
            if a is None or b is None or a.size != b.size:
                problems.append(f'{label}: capture missing at frame {index + 1}')
                break
            if np.array_equal(a, b):
                ok_frames += 1
            else:
                problems.append(f'{label}: frame {index + 1} differs '
                                f'({int(np.count_nonzero(a != b))} bytes)')
                break
        rows.append({'check': label, 'frames_identical': ok_frames,
                     'frames_checked': max(frames - 1, 0),
                     'pass': ok_frames == max(frames - 1, 0) and frames > 1})


def in_frame_edges(dirs, problems, rows):
    """dec0 -> post within one frame: three surfaces cross the stage boundary."""
    dec0 = dirs.get('dec0', [])
    post = dirs.get('post_3_1', [])
    if not dec0 or not post:
        return
    producer = images_of(dec0[0])
    consumer = images_of(post[0])
    consumer_arrays = {image['array'] for image in consumer.values()
                       if image['kind'] == 'tex'}
    confirmed = 0
    for image_id, image in sorted(producer.items()):
        if image['kind'] != 'surf' or image['array'] not in consumer_arrays:
            continue
        a = read_image(dec0[0], image_id, post=True)
        match = next(((cid, cimage) for cid, cimage in consumer.items()
                      if cimage['array'] == image['array']), None)
        b = read_image(post[0], match[0]) if match else None
        same = a is not None and b is not None and a.size == b.size and np.array_equal(a, b)
        confirmed += bool(same)
        if not same:
            problems.append(f'dec0 surface 0x{image["array"]:x} != post input')
    rows.append({'check': 'dec0 -> post image edges', 'confirmed': confirmed,
                 'expected': 3, 'pass': confirmed == 3})
    if confirmed != 3:
        problems.append(f'dec0 -> post: {confirmed} of 3 image edges confirmed')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture')
    ap.add_argument('--outputs', required=True, help="the probe's D4R_DIAG_DIR")
    ap.add_argument('--frames', type=int, default=2,
                    help='frames to check for temporal identity (default 2)')
    ap.add_argument('--json', type=argparse.FileType('w'), default=None)
    a = ap.parse_args()

    dirs = stage_dirs(a.capture)
    problems, rows = [], []
    for stage in ('enc0', 'dec0', 'post_3_1', 'downsample'):
        if not dirs.get(stage):
            problems.append(f'{stage}: no capture directory')
    if dirs.get('enc0'):
        check_enc0(dirs['enc0'][0], problems, rows)
    if dirs.get('dec0'):
        check_dec0(dirs['dec0'][0], problems, rows)
    if dirs.get('post_3_1'):
        check_post(dirs['post_3_1'][0], problems, rows)
    if dirs.get('downsample'):
        check_downsample(dirs['downsample'][0], a.outputs, 1, problems, rows)
    in_frame_edges(dirs, problems, rows)
    if a.frames > 1:
        temporal_edges(a.capture, dirs, a.frames, problems, rows)

    for row in rows:
        state = ('PASS' if row['pass'] else 'FAIL') if row.get('pass') is not None else 'N/A'
        detail = ' '.join(f'{k}={v}' for k, v in row.items()
                          if k not in ('check', 'pass'))
        print(f'{state}  {row["check"]:52s} {detail}')
    print(f'\n{len(problems)} problem(s)')
    for problem in problems:
        print(f'  {problem}')
    if a.json:
        json.dump({'capture': a.capture, 'checks': rows, 'problems': problems,
                   'pass': not problems}, a.json, indent=2)
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
