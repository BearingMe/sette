#!/usr/bin/env python3
"""M05 (B1/C16): audit what a captured frame actually contains, and rebuild the
stage graph from addresses rather than from assumptions.

For every dumped launch this reads the pre-launch allocation images and the
post-launch `gpu-output/` copies, and reports:

  * the parameter block size and grid the hook recorded;
  * which of the launch's pointers resolve into a dumped allocation, and where;
  * which allocations the kernel *wrote* (bytes that differ after the launch),
    with the exact changed ranges;
  * which large allocations are shared with other stages, and at which offsets.

Addresses that alias across stages give the graph's edges.  Allocations too
small to hold pixels, or holding host pointers, are descriptors: they identify a
texture or surface binding without containing its contents, so a texture-bound
image is *not* recoverable from this capture even though its descriptor is
present.  The audit states that explicitly instead of assuming the data is there.

usage: m_capture_audit.py CAPTURE [--json OUT] [--graph]
"""
import argparse
import glob
import json
import os
import re
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import m_weights as mw      # noqa: E402

DESCRIPTOR_LIMIT = 4096        # a binding descriptor, not a tensor


def allocations(directory):
    """id -> {'base', 'size', 'pre', 'post', 'path'} in manifest order."""
    table = {}
    for line in open(os.path.join(directory, 'manifest.txt')):
        q = line.split()
        if q and q[0] == 'alloc':
            index = int(q[1])
            path = os.path.join(directory, f'alloc-{index}.bin')
            post_path = os.path.join(directory, 'gpu-output', f'alloc-{index}.bin')
            table[index] = {
                'base': int(q[2], 16),
                'size': int(q[3]) if len(q) > 3 else os.path.getsize(path),
                'pre': path,
                'post': post_path if os.path.exists(post_path) else None,
                'pre_bytes': os.path.getsize(path),
                'post_bytes': os.path.getsize(post_path) if os.path.exists(post_path) else None,
            }
    return table


def runs(mask):
    """Contiguous True runs of a boolean mask as (start, stop) half-open pairs."""
    if not mask.any():
        return []
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.nonzero(edges == 1)[0]
    stops = np.nonzero(edges == -1)[0]
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def coalesce(spans, gap=512):
    """Merge spans separated by fewer than `gap` unchanged bytes.

    Written tensor regions are dense; a handful of byte-identical values inside
    one buffer would otherwise split it into thousands of ranges.  The merge is
    only used to describe which buffer a stage wrote - the changed-byte count is
    still exact - so a sub-kilobyte gap cannot hide a separate tensor.
    """
    out = []
    for start, stop in spans:
        if out and start - out[-1][1] <= gap:
            out[-1][1] = stop
        else:
            out.append([start, stop])
    return [(a, b) for a, b in out]


def changed(directory, entry):
    """Byte ranges the kernel wrote into this allocation."""
    if not entry['post']:
        return None
    pre = np.fromfile(entry['pre'], np.uint8)
    post = np.fromfile(entry['post'], np.uint8)
    if pre.size != post.size:
        return {'size_mismatch': [int(pre.size), int(post.size)]}
    if pre.size > 64 * 1024 * 1024:
        return {'too_large': int(pre.size)}
    mask = pre != post
    spans = runs(mask)
    merged = coalesce(spans)
    return {'changed_bytes': int(mask.sum()),
            'regions': [[a, b] for a, b in merged],
            'region_count': len(merged),
            'raw_region_count': len(spans),
            'first': int(np.nonzero(mask)[0][0]) if mask.any() else None,
            'last': int(np.nonzero(mask)[0][-1]) if mask.any() else None}


def host_pointers(path, limit=8):
    """Whether the block holds plausible user-space pointers (a descriptor)."""
    data = open(path, 'rb').read()
    if len(data) % 8:
        return 0
    values = struct.unpack('<%dQ' % (len(data) // 8), data)
    hits = [v for v in values if 0x10000 <= v <= 0x7FFFFFFFFFFF and v % 4 == 0]
    return len(hits[:limit])


# ---------------------------------------------------------------------------
# image resources (textures and surfaces)


def images(directory):
    """Declared image targets of one launch, with their bindings.

    Reads the `image` / `image-bind` / `image-unknown` lines the capture hook
    writes next to `alloc`/`pointer`.  A target is one backing array or pitched
    allocation, so the same array bound as a texture and as a surface appears
    once with two bindings - that identity is what makes aliasing visible.
    """
    table = {}
    bindings = []
    unknown = []
    with open(os.path.join(directory, 'manifest.txt')) as handle:
        for line in handle:
            q = line.split()
            if not q:
                continue
            if q[0] == 'image':
                table[int(q[1])] = {
                    'kind': q[2], 'handle': int(q[3], 16), 'array': int(q[4], 16),
                    'pitched': int(q[5], 16), 'width': int(q[6]),
                    'height': int(q[7]), 'depth': int(q[8]),
                    'format': int(q[9], 16), 'channels': int(q[10]),
                    'original': int(q[11]), 'row_bytes': int(q[12]),
                    'bytes': int(q[13]), 'pitch': int(q[14]) if len(q) > 14 else 0,
                    'bindings': [],
                }
            elif q[0] == 'image-bind':
                bindings.append({'image': int(q[1]), 'offset': int(q[2]), 'kind': q[3]})
            elif q[0] == 'image-unknown':
                unknown.append({'offset': int(q[1]), 'handle': q[2],
                                'reason': ' '.join(q[3:])})
    for binding in bindings:
        if binding['image'] in table:
            table[binding['image']]['bindings'].append(binding)
    return table, unknown


def image_state(directory, image_id):
    """Pre/post bytes of one image target, and whether the launch changed it.

    Returns None when the capture is missing a file.  `wrote` is the only
    evidence of an output role used here: a surface handle is not proof that a
    stage wrote, and a changed image is.
    """
    entry = {'pre': os.path.join(directory, f'image-{image_id}.bin'),
             'post': os.path.join(directory, 'gpu-output', f'image-{image_id}.bin')}
    for key in ('pre', 'post'):
        if not os.path.exists(entry[key]):
            return None
    pre = np.fromfile(entry['pre'], np.uint8)
    post = np.fromfile(entry['post'], np.uint8)
    if pre.size != post.size:
        return {'size_mismatch': [int(pre.size), int(post.size)]}
    mask = pre != post
    return {'bytes': int(pre.size), 'changed_bytes': int(mask.sum()),
            'wrote': bool(mask.any()),
            'first': int(np.nonzero(mask)[0][0]) if mask.any() else None,
            'last': int(np.nonzero(mask)[0][-1]) if mask.any() else None}



def audit(capture):
    stages = mw.discover(capture)
    seen = {}
    rows = []
    for stage in stages:
        seen[stage.stage] = seen.get(stage.stage, 0) + 1
        entry = {'stage': stage.stage, 'kernel': stage.kernel,
                 'instance': seen[stage.stage], 'sequence': stage.sequence,
                 'rank': stage.position, 'variant': stage.variant,
                 'grid': None, 'args_bytes': None, 'allocations': [], 'pointers': [],
                 'images': [], 'image_unknown': []}
        with open(os.path.join(stage.directory, 'manifest.txt')) as handle:
            lines = handle.read().split('\n')
        launch = lines[1].split()
        entry['grid'] = [int(v) for v in launch[1:4]]
        entry['block'] = [int(v) for v in launch[4:7]]
        entry['args_bytes'] = int(lines[2].split()[1])
        entry['directory'] = stage.directory
        entry['images'], entry['image_unknown'] = images(stage.directory)
        table = allocations(stage.directory)
        shared = {}
        for index, alloc in sorted(table.items()):
            diff = changed(stage.directory, alloc)
            kind = 'tensor'
            if alloc['pre_bytes'] <= DESCRIPTOR_LIMIT:
                kind = 'descriptor' if host_pointers(alloc['pre']) else 'small'
            elif alloc['pre_bytes'] >= 8 * 1024 * 1024:
                kind = 'large'
            entry['allocations'].append({
                'id': index, 'base': hex(alloc['base']), 'size': alloc['pre_bytes'],
                'kind': kind,
                'written': bool(diff and diff.get('changed_bytes')),
                'diff': diff,
            })
            shared.setdefault(alloc['base'], []).append((stage.stage, index))
        entry['pointers'] = [{'offset': offset, 'address': hex(address)}
                             for offset, address in mw.pointer_addresses(stage.directory)]
        rows.append(entry)
    return rows


def graph(rows):
    """Report which captured buffers each stage wrote (address ranges, coalesced)."""
    written = []
    for entry in rows:
        for alloc in entry['allocations']:
            if not alloc['written'] or 'regions' not in (alloc['diff'] or {}):
                continue
            base = int(alloc['base'], 16)
            for start, stop in alloc['diff']['regions']:
                written.append({'stage': entry['stage'], 'instance': entry['instance'],
                                'rank': entry['rank'], 'base': base,
                                'range': [base + start, base + stop],
                                'bytes': stop - start, 'alloc': alloc['id']})
    return written


def _image(directory, alloc_id, post):
    path = (os.path.join(directory, 'gpu-output', f'alloc-{alloc_id}.bin') if post
            else os.path.join(directory, f'alloc-{alloc_id}.bin'))
    return np.fromfile(path, np.uint8) if os.path.exists(path) else None


def _arena(stage):
    """(allocation id, base, pre image) of the largest allocation a stage touches."""
    table = allocations(stage.directory)
    best = max(table.items(), key=lambda item: item[1]['pre_bytes'])
    index, info = best
    return index, info['base'], np.fromfile(info['pre'], np.uint8)


def edge_checks(capture, rows, written):
    """Confirm every graph edge by tensor identity, not by address coincidence.

    A shared arena means several stages write the same bytes at different times,
    and a parameter block contains output pointers too, so address overlap alone
    proves nothing.  Instead every stage's candidate *output* buffers (its token
    field and, where it has one, its patch-merge field) are matched against every
    other stage's candidate *input* buffers (its token field and skip field) by
    address, and each match must then hold identical bytes: post-launch on the
    producer, pre-launch on the consumer.

    Roles come from the inspected sources: a layer writes `out` and its
    patch-merge result to `p48`, reads `inp` and a skip from `p32`
    (`layer_m.comp`); dec0 reads the token field dec1 wrote (`dec0_m.comp`
    binding 7); post writes the buffer downsample reads.  Sizes are the stages'
    own token geometry, so a match that also agrees on size is not accidental.
    """
    import model_enc3 as me
    stages = mw.discover(capture)
    by_stage = {s.stage: s for s in stages}
    ordered = [s for s in stages if s.variant['family'] == 'layer']
    params = [me.Params(s.directory) for s in ordered]
    outputs, inputs = [], []

    for stage, P in zip(ordered, params):
        C, _, NPM, CIN = mw.mm.LAYERS[stage.stage]
        outputs.append((stage, 'token out', P.out, P.W * P.H * C, stage.position))
        if NPM:
            outputs.append((stage, 'merge out', P.p48, (P.W // 2) * (P.H // 2) * NPM,
                            stage.position))
        size_in = ((P.W // 2) * (P.H // 2) * CIN) if CIN else (P.W * P.H * C)
        inputs.append((stage, 'token in', P.inp, size_in, stage.position))
        if P.p32:
            inputs.append((stage, 'skip in', P.p32, P.W * P.H * C, stage.position))

    enc0_span = next((w for w in written if w['stage'] == 'enc0'), None)
    if enc0_span is not None:
        outputs.append((by_stage['enc0'], 'token out', enc0_span['range'][0],
                        enc0_span['bytes'], by_stage['enc0'].position))

    if ordered and 'dec0' in by_stage:
        shared = ({a for _, a in mw.pointer_addresses(by_stage['dec0'].directory)}
                  & {a for _, a in mw.pointer_addresses(ordered[-1].directory)})
        if shared:
            last = params[-1]
            inputs.append((by_stage['dec0'], 'token in', min(shared),
                           last.W * last.H * mm_channels('dec1'), by_stage['dec0'].position))

    post_span = next((w for w in written if w['stage'] == 'post_3_1'), None)
    if post_span is not None and 'downsample' in by_stage:
        post = by_stage['post_3_1']
        outputs.append((post, 'recon out', post_span['range'][0], post_span['bytes'],
                        post.position))
        inputs.append((by_stage['downsample'], 'recon in', post_span['range'][0],
                       post_span['bytes'], by_stage['downsample'].position))

    checks = []
    for stage, label_in, address, size, rank in inputs:
        # An address can be reused by several stages at different times (the
        # arena is a pool), so a match must agree on address, size and order.
        matches = [o for o in outputs
                   if o[2] == address and o[3] == size and o[4] < rank]
        if not matches:
            checks.append({'edge': f'? -> {stage.stage}', 'label': label_in,
                           'verified': False,
                           'reason': 'no captured producer writes this address before it'})
            continue
        producer, label_out, _, producer_size, _ = max(matches, key=lambda o: o[4])
        same_size = producer_size == size
        pid, pbase, _ = _arena(producer)
        cid, cbase, _ = _arena(stage)
        post = _image(producer.directory, pid, True)
        pre = _image(stage.directory, cid, False)
        if post is None or pre is None:
            checks.append({'edge': f'{producer.stage} -> {stage.stage}', 'label': label_out,
                           'verified': False, 'reason': 'missing allocation image'})
            continue
        start_p, start_c = address - pbase, address - cbase
        a = post[start_p:start_p + size]
        b = pre[start_c:start_c + size]
        verified = a.size == b.size == size and bool(np.array_equal(a, b))
        checks.append({'edge': f'{producer.stage} -> {stage.stage}',
                       'label': f'{label_out} / {label_in}', 'bytes': int(size),
                       'size_agrees': same_size, 'verified': verified,
                       'differing': int(np.count_nonzero(a != b)) if a.size == b.size else None})
    checks.sort(key=lambda c: (c['edge'], c['label']))
    return checks


def image_edges(rows):
    """Confirm image-mediated edges inside one frame.

    Identity is the backing array (or pitched device pointer) the hook recorded,
    which is a real allocation identity rather than a reused arena offset.  An
    edge still has to be confirmed by content: the producer's post-launch image
    must equal the consumer's pre-launch image byte for byte.  A stage's changed
    images are its outputs, its unchanged images are its inputs - the surface
    handle type alone is not treated as proof of a role.

    A bound image with no in-frame producer is reported as `external`: it comes
    from an earlier frame (history, recurrent feature) or from outside the
    network (colour, depth, motion), and this capture cannot show that.
    """
    written, read = [], []
    missing = []
    for entry in rows:
        for image_id, image in sorted(entry['images'].items()):
            state = image_state(entry['directory'], image_id)
            row = {'stage': entry['stage'], 'instance': entry['instance'],
                   'rank': entry['rank'], 'image': image_id, 'kind': image['kind'],
                   'directory': entry['directory'],
                   'key': image['array'] or image['pitched'],
                   'width': image['width'], 'height': image['height'],
                   'depth': image['depth'], 'format': image['format'],
                   'channels': image['channels'], 'bytes': image['bytes'],
                   'bindings': image['bindings'], 'state': state}
            if state is None or 'size_mismatch' in state:
                missing.append(row)
            elif state['wrote']:
                written.append(row)
            else:
                read.append(row)

    edges = []
    for consumer in read:
        matches = [p for p in written if p['key'] == consumer['key']
                   and p['rank'] < consumer['rank']]
        if not matches:
            edges.append({'edge': f"? -> {consumer['stage']}", 'image': consumer['image'],
                          'label': f"{consumer['kind']} 0x{consumer['key']:x} at arg "
                                   f"{consumer['bindings'][0]['offset'] if consumer['bindings'] else '?'}",
                          'verified': None,
                          'reason': 'no in-frame producer: earlier frame or external input'})
            continue
        producer = max(matches, key=lambda p: p['rank'])
        a = np.fromfile(os.path.join(producer['directory'], 'gpu-output',
                                     f"image-{producer['image']}.bin"), np.uint8)
        b = np.fromfile(os.path.join(consumer['directory'],
                                     f"image-{consumer['image']}.bin"), np.uint8)
        differing = int(np.count_nonzero(a != b)) if a.size == b.size else None
        edges.append({'edge': f"{producer['stage']}#{producer['instance']} -> "
                              f"{consumer['stage']}#{consumer['instance']}",
                      'image': consumer['image'],
                      'label': f"{consumer['kind']} 0x{consumer['key']:x}",
                      'bytes': int(b.size),
                      'verified': bool(a.size == b.size and differing == 0),
                      'differing': differing})
    edges.sort(key=lambda e: (e['edge'], e['label']))
    return edges, missing


def mm_channels(stage):
    return {'enc1': 64, 'enc2': 96, 'enc3_tube': 128, 'dec2': 96, 'dec1': 64}[stage]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture')
    ap.add_argument('--json', type=argparse.FileType('w'), default=None)
    ap.add_argument('--graph', action='store_true', help='print only the resolved graph')
    a = ap.parse_args()

    rows = audit(a.capture)
    written = graph(rows)
    checks = edge_checks(a.capture, rows, written)

    if not a.graph:
        print('B1 capture audit')
        for entry in rows:
            variant = (f"{entry['variant']['motion']}/{entry['variant']['range']}"
                       if entry['variant']['motion'] else '')
            wrote = [x for x in entry['allocations'] if x['written']]
            kinds = {}
            for x in entry['allocations']:
                kinds[x['kind']] = kinds.get(x['kind'], 0) + 1
            print(f"  {entry['stage']:10s} {entry['kernel']:36s} {variant:10s} "
                  f"grid={entry['grid']} block={entry['block']} args={entry['args_bytes']:3d}B "
                  f"allocs={len(entry['allocations'])} {kinds} written={len(wrote)}")
            for x in wrote:
                diff = x['diff']
                if 'regions' not in diff:
                    print(f'      alloc {x["id"]}: {diff}')
                    continue
                total = diff['changed_bytes']
                print(f'      alloc {x["id"]} {x["size"]:9d}B: {total:9d}B written in '
                      f'{diff["region_count"]} range(s), first [{diff["first"]}, {diff["last"]}]')
            missing = [x for x in entry['allocations'] if x['kind'] == 'descriptor']
            if missing:
                print(f'      {len(missing)} descriptor binding(s): '
                      f'{len(entry["images"])} image(s) captured, '
                      f'{len(entry["image_unknown"])} unreadable')
            for image_id, image in sorted(entry['images'].items()):
                state = image_state(entry['directory'], image_id)
                slots = ','.join(str(b['offset']) for b in image['bindings'])
                shape = f'{image["width"]}x{image["height"]}'
                if image['depth'] > 1:
                    shape += f'x{image["depth"]}'
                if state is None:
                    print(f'      image {image_id} {shape} fmt=0x{image["format"]:02x} '
                          f'ch={image["channels"]} at {slots}: NOT CAPTURED')
                    continue
                role = f'{state["changed_bytes"]} B changed' if state['wrote'] else 'unchanged'
                print(f'      image {image_id} {image["kind"]:4s} {shape} '
                      f'fmt=0x{image["format"]:02x} ch={image["channels"]} '
                      f'row={image["row_bytes"]} at {slots}: {role}')
            for unknown in entry['image_unknown']:
                print(f'      image at arg {unknown["offset"]} handle {unknown["handle"]}: '
                      f'UNREADABLE {unknown["reason"]}')

    print('\nwritten buffers (coalesced address ranges)')
    for span in written:
        print(f'  {span["stage"]}#{span["instance"]:2d} alloc {span["alloc"]} '
              f'[{hex(span["range"][0])}, {hex(span["range"][1])}) {span["bytes"]:9d} B')

    print('\nedge checks (producer post-launch tensor vs consumer pre-launch tensor)')
    for check in checks:
        if 'reason' in check:
            print(f'  {check["edge"]:26s} {check["label"]:22s} UNVERIFIED: {check["reason"]}')
            continue
        state = 'identical' if check['verified'] else f'{check["differing"]} bytes differ'
        print(f'  {check["edge"]:26s} {check["label"]:22s} {check["bytes"]:9d} B  {state}')
    verified = sum(1 for c in checks if c.get('verified'))
    print(f'  {verified}/{len(checks)} edges confirmed by content')

    print('\nstages whose operands are not captured buffers')
    layers = {s.stage for s in mw.discover(a.capture) if s.variant['family'] == 'layer'}
    for entry in rows:
        if entry['stage'] in layers or entry['stage'] == 'dec0':
            continue
        print(f'  {entry["stage"]}: image/texture bindings only '
              f'({sum(1 for x in entry["allocations"] if x["kind"] == "descriptor")} descriptor(s))')

    edges_image, missing_images = image_edges(rows)
    print('\nimage edge checks (same backing array, producer post vs consumer pre)')
    for check in edges_image:
        if check['verified'] is None:
            print(f'  {check["edge"]:26s} {check["label"]:40s} EXTERNAL: {check["reason"]}')
            continue
        state = 'identical' if check['verified'] else f'{check["differing"]} bytes differ'
        print(f'  {check["edge"]:26s} {check["label"]:40s} {check["bytes"]:9d} B  {state}')
    confirmed = sum(1 for c in edges_image if c.get('verified'))
    external = sum(1 for c in edges_image if c.get('verified') is None)
    print(f'  {confirmed} image edge(s) confirmed by content, {external} external/temporal')
    if missing_images:
        print('  images declared but not captured:')
        for row in missing_images:
            print(f'    {row["stage"]}#{row["instance"]} image {row["image"]} '
                  f'{row["width"]}x{row["height"]}')

    if a.json:
        json.dump({'capture': a.capture, 'stages': rows, 'written': written,
                   'edges': checks, 'image_edges': edges_image,
                   'images_missing': missing_images}, a.json, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
