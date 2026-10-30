#!/usr/bin/env python3
"""Bind each captured launch to its actual logged backend; never infer from source."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def dispatches(text):
    rows = []
    pattern = r'\[d4r-launch\] kernel="([^"]+)" backend=(native|translated) grid=([\d,]+) block=([\d,]+)'
    for line, value in enumerate(text.splitlines(), 1):
        match = re.search(pattern, value)
        if match:
            name, backend, grid, block = match.groups()
            rows.append({'kernel': name, 'backend': backend, 'grid': grid, 'block': block, 'log_line': line})
    return rows


def verify_translated(text, kernel):
    rows = [r for r in dispatches(text) if r['kernel'] == kernel]
    if len(rows) != 1 or rows[0]['backend'] != 'translated':
        raise ValueError('expected exactly one translated dispatch and no native dispatch')
    if f'module_direct kernel="{kernel}"' in text:
        raise ValueError('native module_direct contradicts translated control')
    return rows[0]


def audit(text, captures):
    rows = [r for r in dispatches(text) if r['kernel'].startswith('rrlite_')]
    directories = sorted(p for p in Path(captures).glob('replay-*') if p.is_dir())
    if len(rows) != len(directories) or not rows:
        raise ValueError('capture/dispatch count mismatch or absent dispatch evidence')
    begins = list(re.finditer(r'\[d4r-replay\] begin kernel=(rrlite_\S+)', text))
    if len(begins) != len(rows):
        raise ValueError('missing/extra capture begin records')
    used = set()
    for index, row in enumerate(rows):
        name = row['kernel']
        segment = text[begins[index].start():begins[index + 1].start() if index + 1 < len(begins) else len(text)]
        compact = re.sub(r'\s+', '', segment)
        completion = compact.split('[d4r-replay]completepath=', 1)
        matching = [p for p in directories if len(completion) == 2 and p.name in completion[1]]
        if (begins[index].group(1) != name or len(dispatches(segment)) != 1 or len(matching) != 1):
            raise ValueError('missing capture completion evidence')
        directory = matching[0]
        if directory in used:
            raise ValueError('duplicate capture completion')
        used.add(directory)
        manifest = (directory / 'manifest.txt').read_text()
        if manifest.splitlines()[0].split()[1] != name:
            raise ValueError('capture kernel mismatch')
        row['capture'] = directory.name
        row['manifest_sha256'] = hashlib.sha256(manifest.encode()).hexdigest()
        row['native_module_direct'] = f'module_direct kernel="{name}"' in text
        if row['backend'] == 'translated' and row['native_module_direct']:
            raise ValueError('contradictory translated/native evidence')
        row['numerical_mode'] = 'not established by dispatch log' if row['backend'] == 'native' else 'ZLUDA-translated PTX'
        if row['backend'] == 'native':
            compact_log = re.sub(r'\s+', '', text)
            match = re.search(r'\[d4r-native\]"' + re.escape(name) + r'"->(.+?\.hsaco)', compact_log)
            if not match:
                raise ValueError('native module path not logged')
            module = Path(match.group(1))
            row['logged_module'] = str(module)
            # Audit-time hashes cannot establish that the file was unchanged since capture.
            row['module_sha256_at_audit'] = hashlib.sha256(module.read_bytes()).hexdigest() if module.is_file() else None
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--log', type=Path, required=True)
    ap.add_argument('--capture', type=Path, required=True)
    ap.add_argument('--json', type=Path, required=True)
    a = ap.parse_args()
    raw = a.log.read_bytes()
    text = raw.decode('utf-16') if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else raw.decode('utf-8')
    report = {'log': str(a.log), 'log_sha256': hashlib.sha256(raw).hexdigest(), 'launches': audit(text, a.capture)}
    a.json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
