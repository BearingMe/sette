#!/usr/bin/env python3
"""B7 readiness: what the existing multi-frame capture does and does not contain.

B7 is temporal validation.  Before implementing any of it, this answers the
narrow question the roadmap needs: which temporal resources are captured, with
what geometry and format, and whether a producer/consumer handoff is directly
observable between replays.

The capture dumps the storage behind every image a launch binds, plus the launch
manifest (per-image geometry, format code, pitch, byte count and parameter
offset).  Nothing here is inferred from a shader: a role is only listed when the
manifest is present, and the content claims come from hashing the dumped bytes.

  m_b7_readiness.py CAPTURE_DIR [--json OUT] [--frame N ...]

`CAPTURE_DIR` is a directory of `replay-NNNNNN-<kernel>` capture folders (the
image capture, `capture-img`).  The role names come from the stage contracts
already established: `param+56`/`param+96` for `post` (EXP-016), `param+56`/
`param+64`/`param+128` for `dec0` (EXP-014), `param+96`/`param+280`/`288`/`296`
for `enc0` (EXP-015).  A role whose offset is absent is reported absent.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

#: kernel -> parameter offset -> (role, expected image kind)
ROLES = {
    'post_3_1_mvhi_ldr_folded': {
        56: ('history_in', 'tex'),
        96: ('history_out', 'surf'),
        112: ('half_out', 'surf'),
    },
    'dec0_4x4_folded': {
        56: ('feature_in_f16', 'tex'),
        64: ('feature_in_snorm8', 'tex'),
        128: ('feature_out_snorm8', 'surf'),
    },
    'enc0_4x4_mvhi_ldr_folded': {
        96: ('previous_motion_in', 'tex'),
        280: ('motion_a_out', 'surf'),
        288: ('motion_b_out', 'surf'),
        296: ('motion_c_out', 'surf'),
    },
}

#: producer -> consumer expressions that B7 needs to hold between consecutive
#: frames, keyed (kernel, producer role, kernel, consumer role).
HANDOFFS = [
    ('post_3_1_mvhi_ldr_folded', 'history_out', 'post_3_1_mvhi_ldr_folded', 'history_in'),
    ('dec0_4x4_folded', 'feature_out_snorm8', 'dec0_4x4_folded', 'feature_in_snorm8'),
    ('enc0_4x4_mvhi_ldr_folded', 'motion_a_out', 'enc0_4x4_mvhi_ldr_folded',
     'previous_motion_in'),
]

NAME = re.compile(r'replay-(\d+)-rrlite_(.+)$')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_manifest(path):
    """{offset: {kind, image, width, height, format, pitch, bytes}} from a capture."""
    images, binds, kernel = {}, {}, None
    for line in Path(path).read_text().splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == 'kernel':
            name = words[1]
            kernel = name[len('rrlite_'):] if name.startswith('rrlite_') else name
        elif words[0] == 'image':
            index = int(words[1])
            images[index] = {'kind': words[2], 'width': int(words[6]),
                             'height': int(words[7]), 'format': words[9],
                             'components': int(words[10]), 'pitch': int(words[12]),
                             'bytes': int(words[13]), 'file': f'image-{index}.bin'}
        elif words[0] == 'image-bind':
            binds[int(words[1])] = (int(words[2]), words[3])
    out = {}
    for index, offset_kind in {i: b for i, b in binds.items()}.items():
        offset, kind = offset_kind
        if index not in images:
            continue
        row = dict(images[index])
        row['kind'] = kind
        out[offset] = row
    return kernel, out


def scan(capture):
    """{kernel: {frame: {offset: row with content statistics}}}"""
    found = {}
    for directory in sorted(Path(capture).iterdir()):
        if not directory.is_dir():
            continue
        match = NAME.search(directory.name)
        if not match:
            continue
        frame, kernel = int(match.group(1)), match.group(2)
        manifest = directory / 'manifest.txt'
        if not manifest.is_file():
            continue
        _, images = parse_manifest(manifest)
        for offset, row in images.items():
            path = directory / row['file']
            if not path.is_file():
                row['present'] = False
                continue
            data = path.read_bytes()
            row['present'] = True
            row['sha256'] = hashlib.sha256(data).hexdigest()
            row['bytes_on_disk'] = len(data)
            row['nonzero_bytes'] = sum(1 for b in data if b)
            row['all_zero'] = row['nonzero_bytes'] == 0
        found.setdefault(kernel, {})[frame] = images
    return found


def audit(capture):
    found = scan(capture)
    report = {'capture': str(capture), 'kernels': {}, 'handoffs': [],
              'missing_roles': [], 'frames': {}}
    for kernel, frames in found.items():
        report['frames'][kernel] = sorted(frames)
        roles = ROLES.get(kernel, {})
        entry = {'replays': sorted(frames), 'roles': {}}
        for offset, (role, kind) in roles.items():
            rows = {}
            for frame in sorted(frames):
                row = frames[frame].get(offset)
                rows[frame] = ({k: row[k] for k in
                                ('kind', 'width', 'height', 'format', 'components',
                                 'bytes', 'present', 'sha256', 'all_zero',
                                 'nonzero_bytes') if k in row} if row else {'present': False})
            distinct = {v.get('sha256') for v in rows.values() if v.get('present')}
            entry['roles'][role] = {
                'offset': offset, 'expected_kind': kind,
                'formed': sorted(rows),
                'geometry': sorted({f"{v.get('width')}x{v.get('height')}"
                                    for v in rows.values() if v.get('present')}),
                'bytes': sorted({v.get('bytes') for v in rows.values() if v.get('present')}),
                'distinct_contents': len(distinct),
                'per_frame': rows,
            }
            if any(not v.get('present') for v in rows.values()):
                report['missing_roles'].append({'kernel': kernel, 'role': role})
        report['kernels'][kernel] = entry

    for p_kernel, p_role, c_kernel, c_role in HANDOFFS:
        producer = report['kernels'].get(p_kernel, {}).get('roles', {}).get(p_role)
        consumer = report['kernels'].get(c_kernel, {}).get('roles', {}).get(c_role)

        def filled(role):
            """A role exists only if at least one replay actually bound its image."""
            return bool(role) and any(row.get('present') for row in
                                      (role.get('per_frame') or {}).values())

        row = {'producer': f'{p_kernel}.{p_role}', 'consumer': f'{c_kernel}.{c_role}',
               'observed': filled(producer) and filled(consumer),
               'equal_pairs': [], 'size_match': None}
        if row['observed']:
            row['size_match'] = producer['bytes'] == consumer['bytes']
            for pf, prow in (producer.get('per_frame') or {}).items():
                for cf, crow in (consumer.get('per_frame') or {}).items():
                    if not (prow.get('present') and crow.get('present')):
                        continue
                    if prow['sha256'] == crow['sha256']:
                        row['equal_pairs'].append({'producer_frame': pf,
                                                   'consumer_frame': cf,
                                                   'all_zero': prow.get('all_zero')})
        else:
            missing = [name for name, value in (('producer', producer),
                                                ('consumer', consumer)) if not filled(value)]
            row['missing'] = missing
        report['handoffs'].append(row)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    report = audit(a.capture)
    for kernel, entry in report['kernels'].items():
        print(f'{kernel}: replays {entry["replays"]}')
        for role, row in entry['roles'].items():
            zeros = [f for f, v in row['per_frame'].items() if v.get('all_zero')]
            print(f"  {role:24s} {row['geometry']} {row['bytes']} B, "
                  f"{row['distinct_contents']} distinct content(s), zero on {zeros or 'none'}")
    for row in report['handoffs']:
        print(f"{row['producer']} -> {row['consumer']}: observed={row['observed']} "
              f"size_match={row['size_match']} equal={row['equal_pairs']}")
    if report['missing_roles']:
        print('missing roles:', report['missing_roles'])
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
