#!/usr/bin/env python3
"""M05 (A2/A3): prove the replay hook's image capture round-trips known images.

The capture extension records the storage behind every texture/surface object a
launch binds.  This test drives the *real* hook through the *real* driver API
with synthetic arrays whose bytes are known independently:

  * a 2D RGBA16F array (37x13, 296-byte rows, uploaded with a padded host pitch),
  * a 2D UNORM_INT8X4 array (5x7, 20-byte rows) that is bound as a texture *and*
    as a surface, so the alias path is exercised,
  * a 3D HALF array (3x4x2), which exercises the 3D readback.

A trivial kernel named `rrlite_enc1_4x4` (a name the hook whitelists) binds the
objects and returns.  Acceptance criteria, fixed before execution:

  1. every bound object appears in the manifest as an `image` line with the
     dimensions, format and row pitch it was created with;
  2. the pre-launch image bytes equal the uploaded pattern byte for byte,
     including the padded-pitch upload and the 3D array;
  3. the texture and the surface on one array are a single target with two
     bindings (alias identity, not duplication);
  4. the post-launch capture is identical to the pre-launch capture, because
     this kernel writes nothing (an input must not be disturbed by capturing it);
  5. two runs produce byte-identical image contents;
  6. no `image-unknown` line is emitted, i.e. every binding was readable.

Read/write *roles* are not established here: this driver's PTX lowering rejects
the hand-written surface store forms tried (see the experiment report), and a
real NVIDIA stage-written surface is a better witness anyway - `dec0`'s four
surfaces are checked in the frame capture.  What this test fixes is the storage
contract: geometry, format, pitch, byte extent and readback fidelity.

usage: m_image_capture_test.py [--cuda-dll PATH] [--keep]
"""
import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from ctypes import (POINTER, Structure, byref, c_char_p, c_float, c_int,
                    c_size_t, c_uint, c_uint64, c_void_p)

DEFAULT_CUDA = os.path.join(tempfile.gettempdir(), 'opencode', 'd4r-runtime', 'd4r',
                            'zluda-c10', 'nvcuda.dll')

CUDA_SUCCESS = 0
CU_AD_FORMAT_UNSIGNED_INT8 = 0x01
CU_AD_FORMAT_HALF = 0x10
CU_AD_FORMAT_UNORM_INT8X4 = 194
CU_RESOURCE_TYPE_ARRAY = 0x00

CASES = [
    # name, width, height, depth, format, channels, 3d
    ('rgba16f', 37, 13, 1, CU_AD_FORMAT_HALF, 4, False),
    ('u8x4', 5, 7, 1, CU_AD_FORMAT_UNORM_INT8X4, 4, False),
    ('half3d', 3, 4, 2, CU_AD_FORMAT_HALF, 1, True),
]

KERNEL_PTX = b"""
.version 8.0
.target sm_80
.address_size 64
.visible .entry rrlite_enc1_4x4(.param .align 8 .b8 args[56])
{
    ret;
}
""" + b'\0'


class CUDA_ARRAY_DESCRIPTOR(Structure):
    _fields_ = [('Width', c_size_t), ('Height', c_size_t),
                ('Format', c_uint), ('NumChannels', c_uint)]


class CUDA_ARRAY3D_DESCRIPTOR(Structure):
    _fields_ = [('Width', c_size_t), ('Height', c_size_t), ('Depth', c_size_t),
                ('Format', c_uint), ('NumChannels', c_uint), ('Flags', c_uint)]


class CUDA_MEMCPY2D(Structure):
    _fields_ = [
        ('srcXInBytes', c_size_t), ('srcY', c_size_t),
        ('srcMemoryType', c_uint), ('srcHost', c_void_p), ('srcDevice', c_uint64),
        ('srcArray', c_void_p), ('srcPitch', c_size_t),
        ('dstXInBytes', c_size_t), ('dstY', c_size_t),
        ('dstMemoryType', c_uint), ('dstHost', c_void_p), ('dstDevice', c_uint64),
        ('dstArray', c_void_p), ('dstPitch', c_size_t),
        ('WidthInBytes', c_size_t), ('Height', c_size_t),
    ]


class CUDA_MEMCPY3D(Structure):
    _fields_ = [
        ('srcXInBytes', c_size_t), ('srcY', c_size_t), ('srcZ', c_size_t),
        ('srcLOD', c_size_t), ('srcMemoryType', c_uint), ('srcHost', c_void_p),
        ('srcDevice', c_uint64), ('srcArray', c_void_p), ('reserved0', c_void_p),
        ('srcPitch', c_size_t), ('srcHeight', c_size_t),
        ('dstXInBytes', c_size_t), ('dstY', c_size_t), ('dstZ', c_size_t),
        ('dstLOD', c_size_t), ('dstMemoryType', c_uint), ('dstHost', c_void_p),
        ('dstDevice', c_uint64), ('dstArray', c_void_p), ('reserved1', c_void_p),
        ('dstPitch', c_size_t), ('dstHeight', c_size_t),
        ('WidthInBytes', c_size_t), ('Height', c_size_t), ('Depth', c_size_t),
    ]


class RESOURCE_DESC(Structure):
    """CUDA_RESOURCE_DESC: resType, a 128-byte union, flags."""
    _fields_ = [('resType', c_uint), ('res', c_uint64 * 16),
                ('flags', c_uint), ('reserved', c_uint)]


class TEXTURE_DESC(Structure):
    _fields_ = [('addressMode', c_int * 3), ('filterMode', c_int),
                ('flags', c_uint), ('maxAnisotropy', c_uint),
                ('mipmapFilterMode', c_int), ('mipmapLevelBias', c_float),
                ('minMipmapLevelClamp', c_float), ('maxMipmapLevelClamp', c_float),
                ('borderColor', c_float * 4), ('reserved', c_int * 12)]


def element_bytes(fmt):
    """Storage element width of a CUDA array format.

    Formats 192-203 are D3D-style enums the driver maps to equivalent integer
    storage (`zluda/src/impl/array.rs`); 80 is packed 10:10:10:2, stored RGBA32F.
    """
    if fmt in (0x01, 0x08):
        return 1
    if fmt in (0x02, 0x09, 0x10):
        return 2
    if fmt in (0x03, 0x0a, 0x20):
        return 4
    if fmt == 80:
        return 4
    if 192 <= fmt <= 197:
        return 1 if fmt <= 194 else 2
    if 198 <= fmt <= 203:
        return 1 if fmt <= 200 else 2
    raise ValueError(f'format {fmt}')


def pattern(width, height, depth, element, channels):
    """Deterministic bytes: every element of every channel is distinct."""
    row = width * channels * element
    out = bytearray(row * height * depth)
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                base = (z * height + y) * row + x * channels * element
                for c in range(channels):
                    seed = (x * 37 + y * 59 + z * 83 + c * 101) & 0xFF
                    for b in range(element):
                        out[base + c * element + b] = (seed + 17 * b + 3 * c) & 0xFF
    return np.frombuffer(bytes(out), np.uint8), row


class Driver:
    def __init__(self, path):
        self.lib = ctypes.WinDLL(path)
        lib = self.lib
        lib.cuInit.argtypes = [c_uint]
        lib.cuInit.restype = c_int
        lib.cuDeviceGet.argtypes = [POINTER(c_int), c_int]
        lib.cuDeviceGet.restype = c_int
        lib.cuCtxCreate_v2.argtypes = [POINTER(c_void_p), c_uint, c_int]
        lib.cuCtxCreate_v2.restype = c_int
        lib.cuArrayCreate_v2.argtypes = [POINTER(c_void_p), POINTER(CUDA_ARRAY_DESCRIPTOR)]
        lib.cuArrayCreate_v2.restype = c_int
        lib.cuArray3DCreate_v2.argtypes = [POINTER(c_void_p), POINTER(CUDA_ARRAY3D_DESCRIPTOR)]
        lib.cuArray3DCreate_v2.restype = c_int
        lib.cuMemcpy2D_v2.argtypes = [POINTER(CUDA_MEMCPY2D)]
        lib.cuMemcpy2D_v2.restype = c_int
        lib.cuMemcpy3D_v2.argtypes = [POINTER(CUDA_MEMCPY3D)]
        lib.cuMemcpy3D_v2.restype = c_int
        lib.cuTexObjectCreate.argtypes = [POINTER(c_uint64), POINTER(RESOURCE_DESC),
                                         POINTER(TEXTURE_DESC), c_void_p]
        lib.cuTexObjectCreate.restype = c_int
        lib.cuSurfObjectCreate.argtypes = [POINTER(c_uint64), POINTER(RESOURCE_DESC)]
        lib.cuSurfObjectCreate.restype = c_int
        lib.cuMemAlloc_v2.argtypes = [POINTER(c_uint64), c_size_t]
        lib.cuMemAlloc_v2.restype = c_int
        lib.cuMemFree_v2.argtypes = [c_uint64]
        lib.cuMemFree_v2.restype = c_int
        lib.cuModuleLoadData.argtypes = [POINTER(c_void_p), c_void_p]
        lib.cuModuleLoadData.restype = c_int
        lib.cuModuleGetFunction.argtypes = [POINTER(c_void_p), c_void_p, c_char_p]
        lib.cuModuleGetFunction.restype = c_int
        lib.cuLaunchKernel.argtypes = ([c_void_p] + [c_uint] * 6
                                       + [c_uint, c_void_p, c_void_p, c_void_p])
        lib.cuLaunchKernel.restype = c_int
        lib.cuStreamSynchronize.argtypes = [c_void_p]
        lib.cuStreamSynchronize.restype = c_int

        self.check(lib.cuInit(0), 'cuInit')
        device = c_int()
        self.check(lib.cuDeviceGet(byref(device), 0), 'cuDeviceGet')
        context = c_void_p()
        self.check(lib.cuCtxCreate_v2(byref(context), 0, device), 'cuCtxCreate_v2')

    def check(self, result, what):
        if result != CUDA_SUCCESS:
            raise RuntimeError(f'{what} failed with CUresult {result}')

    def array(self, case):
        name, width, height, depth, fmt, channels, is3d = case
        handle = c_void_p()
        if is3d:
            desc = CUDA_ARRAY3D_DESCRIPTOR(width, height, depth, fmt, channels, 0)
            self.check(self.lib.cuArray3DCreate_v2(byref(handle), byref(desc)),
                       f'cuArray3DCreate {name}')
        else:
            desc = CUDA_ARRAY_DESCRIPTOR(width, height, fmt, channels)
            self.check(self.lib.cuArrayCreate_v2(byref(handle), byref(desc)),
                       f'cuArrayCreate {name}')
        return handle

    def upload_array(self, case, handle, data, row):
        """Upload through a staging buffer whose pitch deliberately exceeds the row."""
        name, width, height, depth, fmt, channels, is3d = case
        host_pitch = row + 32
        staging = np.zeros(host_pitch * height * depth, np.uint8)
        for r in range(height * depth):
            staging[r * host_pitch:r * host_pitch + row] = data[r * row:(r + 1) * row]
        anchor = ctypes.create_string_buffer(staging.tobytes(), staging.size)
        if is3d:
            copy = CUDA_MEMCPY3D()
            copy.srcMemoryType = 1  # HOST
            copy.srcHost = ctypes.cast(anchor, c_void_p).value
            copy.srcPitch = host_pitch
            copy.srcHeight = height
            copy.dstMemoryType = 3  # ARRAY
            copy.dstArray = handle
            copy.WidthInBytes = row
            copy.Height = height
            copy.Depth = depth
            self.check(self.lib.cuMemcpy3D_v2(byref(copy)), f'cuMemcpy3D {name}')
        else:
            copy = CUDA_MEMCPY2D()
            copy.srcMemoryType = 1  # HOST
            copy.srcHost = ctypes.cast(anchor, c_void_p).value
            copy.srcPitch = host_pitch
            copy.dstMemoryType = 3  # ARRAY
            copy.dstArray = handle
            copy.WidthInBytes = row
            copy.Height = height
            self.check(self.lib.cuMemcpy2D_v2(byref(copy)), f'cuMemcpy2D {name}')
        return anchor

    def texture(self, handle):
        resource = RESOURCE_DESC()
        resource.resType = CU_RESOURCE_TYPE_ARRAY
        resource.res[0] = handle.value or 0
        texture_desc = TEXTURE_DESC()
        texture_desc.filterMode = 0            # POINT
        for axis in range(3):
            texture_desc.addressMode[axis] = 1  # CLAMP
        out = c_uint64()
        self.check(self.lib.cuTexObjectCreate(byref(out), byref(resource),
                                              byref(texture_desc), None),
                   'cuTexObjectCreate')
        return out.value

    def surface(self, handle):
        resource = RESOURCE_DESC()
        resource.resType = CU_RESOURCE_TYPE_ARRAY
        resource.res[0] = handle.value or 0
        out = c_uint64()
        self.check(self.lib.cuSurfObjectCreate(byref(out), byref(resource)),
                   'cuSurfObjectCreate')
        return out.value

    def launch(self, block):
        ptx = ctypes.create_string_buffer(KERNEL_PTX, len(KERNEL_PTX))
        module = c_void_p()
        self.check(self.lib.cuModuleLoadData(byref(module), ptx), 'cuModuleLoadData')
        function = c_void_p()
        self.check(self.lib.cuModuleGetFunction(byref(function), module,
                                                b'rrlite_enc1_4x4'),
                   'cuModuleGetFunction')
        parameters = ctypes.create_string_buffer(block, len(block))
        param_array = (c_void_p * 1)(ctypes.cast(parameters, c_void_p))
        self.check(self.lib.cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0,
                                           None, ctypes.cast(param_array, c_void_p), None),
                   'cuLaunchKernel')
        self.check(self.lib.cuStreamSynchronize(None), 'cuStreamSynchronize')


def run_once(driver, directory):
    """One capture: three arrays, one surface alias, one launch."""
    expected, handles, anchors = {}, {}, []
    for case in CASES:
        name = case[0]
        handle = driver.array(case)
        data, row = pattern(case[1], case[2], case[3], element_bytes(case[4]), case[5])
        anchors.append(driver.upload_array(case, handle, data, row))
        expected[name] = {'bytes': data, 'row': row, 'geometry': case[1:4]}
        handles[name] = handle

    block = np.zeros(56, np.uint8)
    textures = {name: driver.texture(handles[name])
                for name in ('rgba16f', 'u8x4', 'half3d')}
    surface = driver.surface(handles['u8x4'])
    scratch = c_uint64()
    driver.check(driver.lib.cuMemAlloc_v2(byref(scratch), 4096), 'cuMemAlloc_v2')
    block[0:40] = np.frombuffer(
        np.array([textures['rgba16f'], textures['u8x4'], textures['half3d'], surface,
                  scratch.value], np.uint64).tobytes(), np.uint8)
    os.environ['D4R_CUDA_REPLAY_DUMP_DIR'] = directory
    os.environ['D4R_CUDA_REPLAY_DUMP_FILTER'] = 'rrlite_'
    os.environ['D4R_CUDA_REPLAY_DUMP_LIMIT'] = '40'
    driver.launch(bytes(block))
    driver.lib.cuMemFree_v2(scratch.value)
    return expected


def read_manifest(directory):
    stage = None
    for name in sorted(os.listdir(directory)):
        candidate = os.path.join(directory, name)
        if name.startswith('replay-') and os.path.isdir(candidate) and \
                name.endswith('rrlite_enc1_4x4'):
            stage = candidate
    if stage is None:
        raise AssertionError(f'no replay directory under {directory}')
    images, binds, unknown = {}, [], []
    for line in open(os.path.join(stage, 'manifest.txt')):
        q = line.split()
        if not q:
            continue
        if q[0] == 'image':
            images[int(q[1])] = {
                'kind': q[2], 'handle': int(q[3], 16), 'array': int(q[4], 16),
                'pitched': int(q[5], 16), 'width': int(q[6]), 'height': int(q[7]),
                'depth': int(q[8]), 'format': int(q[9], 16), 'channels': int(q[10]),
                'original': int(q[11]), 'row': int(q[12]), 'bytes': int(q[13]),
                'pitch': int(q[14]) if len(q) > 14 else 0,
            }
        elif q[0] == 'image-bind':
            binds.append((int(q[1]), int(q[2]), q[3]))
        elif q[0] == 'image-unknown':
            unknown.append((int(q[1]), q[2], ' '.join(q[3:])))
    return stage, images, binds, unknown


def compare(directory, expected):
    """Acceptance checks 1-4 and 6 against the uploaded patterns."""
    stage, images, binds, unknown = read_manifest(directory)
    problems = []
    if unknown:
        problems.append(f'image-unknown lines present: {unknown}')

    by_binding = {}
    for image_id, offset, kind in binds:
        by_binding.setdefault(offset, []).append((image_id, kind))

    if len(by_binding.get(0, [])) != 1 or len(by_binding.get(24, [])) != 1:
        problems.append(f'bindings missing: {sorted(by_binding)}')
    shared = ({image_id for image_id, _ in by_binding.get(8, [])}
              & {image_id for image_id, _ in by_binding.get(24, [])})
    if len(shared) != 1:
        problems.append('the texture and the surface on one array were not one target')

    for offset, name in ((0, 'rgba16f'), (8, 'u8x4'), (16, 'half3d')):
        found = by_binding.get(offset, [])
        if not found or found[0][0] not in images:
            problems.append(f'no image recorded for binding {offset}')
            continue
        entry = found[0][0]
        image = images[entry]
        want = expected[name]
        width, height, depth = want['geometry']
        if (image['width'], image['height'], image['depth']) != (width, height, depth):
            problems.append(f'{name}: geometry {image["width"]}x{image["height"]}x'
                            f'{image["depth"]} != {width}x{height}x{depth}')
        if image['row'] != want['row']:
            problems.append(f'{name}: row {image["row"]} != {want["row"]}')
        if image['bytes'] != want['row'] * height * depth:
            problems.append(f'{name}: byte extent {image["bytes"]} is inconsistent')
        if name == 'u8x4' and not (image['original'] == CU_AD_FORMAT_UNORM_INT8X4
                                   and image['format'] == CU_AD_FORMAT_UNSIGNED_INT8):
            problems.append(f'u8x4: storage metadata {image["format"]}/'
                            f'{image["original"]} does not describe the INT8X4 backing')
        pre = np.fromfile(os.path.join(stage, f'image-{entry}.bin'), np.uint8)
        post = np.fromfile(os.path.join(stage, 'gpu-output', f'image-{entry}.bin'), np.uint8)
        data = want['bytes']
        if pre.size != data.size or not np.array_equal(pre, data):
            differing = (int(np.count_nonzero(pre != data)) if pre.size == data.size
                         else f'{pre.size} vs {data.size} bytes')
            problems.append(f'{name}: captured pre-launch bytes differ from the upload '
                            f'({differing})')
        if post.size != pre.size or not np.array_equal(post, pre):
            problems.append(f'{name}: the post-launch capture differs from the '
                            f'pre-launch capture for a kernel that wrote nothing')
    return problems, images, binds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cuda-dll', default=DEFAULT_CUDA)
    ap.add_argument('--keep', action='store_true', help='keep the capture directories')
    ap.add_argument('--json', type=argparse.FileType('w'), default=None)
    ap.add_argument('--single', metavar='DIR',
                    help=argparse.SUPPRESS)  # one capture; the driver config is per process
    a = ap.parse_args()

    if a.single:
        driver = Driver(a.cuda_dll)
        expected = run_once(driver, a.single)
        problems, images, binds = compare(a.single, expected)
        print(f'{a.single}: {len(images)} image target(s), {len(problems)} problem(s)')
        for problem in problems:
            print(f'  FAIL {problem}')
        if a.json:
            json.dump({'directory': read_manifest(a.single)[0], 'problems': problems,
                       'images': images, 'bindings': binds}, a.json, indent=2)
        return 0 if not problems else 1

    root = tempfile.mkdtemp(prefix='m-image-capture-')
    results = []
    try:
        # The driver reads D4R_CUDA_REPLAY_DUMP_DIR once per process, so each
        # repetition is its own process; otherwise both runs land in one directory.
        for run in (1, 2):
            directory = os.path.join(root, f'capture-{run}')
            os.makedirs(directory)
            report = os.path.join(root, f'run-{run}.json')
            completed = subprocess.run(
                [sys.executable, os.path.abspath(__file__), '--single', directory,
                 '--cuda-dll', a.cuda_dll, '--json', report],
                capture_output=True, text=True)
            sys.stdout.write(completed.stdout)
            if completed.returncode not in (0, 1):
                sys.stderr.write(completed.stderr)
            with open(report) as handle:
                results.append(json.load(handle))

        same = True
        # Only the image contents are compared: descriptor blobs and the manifest
        # legitimately carry this process's device addresses, which change per run.
        for name in sorted(os.listdir(results[0]['directory'])):
            if not name.startswith('image-'):
                continue
            for folder in (results[0]['directory'],
                           os.path.join(results[0]['directory'], 'gpu-output')):
                other = folder.replace(results[0]['directory'], results[1]['directory'])
                left, right = os.path.join(folder, name), os.path.join(other, name)
                if not os.path.exists(right) or open(left, 'rb').read() != open(right, 'rb').read():
                    same = False
                    print(f'  FAIL repeat differs: '
                          f'{os.path.relpath(left, results[0]["directory"])}')
        print(f'repeat image contents byte-identical: {same}')
        ok = all(not r['problems'] for r in results) and same
        if a.json:
            json.dump({'cuda_dll': a.cuda_dll, 'runs': results, 'repeat_identical': same,
                       'pass': ok}, a.json, indent=2)
        print('PASS image capture round-trip' if ok else 'FAIL image capture round-trip')
        return 0 if ok else 1
    finally:
        if not a.keep:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print(f'captures kept under {root}')


if __name__ == '__main__':
    sys.exit(main())
