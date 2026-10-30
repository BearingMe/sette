#!/usr/bin/env python3
"""Bounded arithmetic-only GPU probe. Captured operands and generated files stay local.

The HIP code is a diagnostic, not a Sette execution backend. Each WMMA tile
replicates one dot product in every row/column, removing fragment ambiguity.
"""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import sys
import sqlite3

import numpy as np

import m_enc1_contract as contract
import m_layer_oracle as lo
import m_layer_fixture as lf
from m_enc1_probe import coordinates


PTX = r'''.version 8.7
.target sm_89
.address_size 64
.visible .entry k32_probe_ptx(.param .u64 ip, .param .u64 op) {
.reg .b64 %rd<8>;
.reg .b32 %r<24>;
ld.param.u64 %rd0, [ip];
ld.param.u64 %rd1, [op];
mov.u32 %r0, %ctaid.x;
mov.u32 %r1, %tid.x;
mad.lo.u32 %r2, %r0, 32, %r1;
mul.wide.u32 %rd2, %r2, 32;
add.u64 %rd3, %rd0, %rd2;
ld.global.v4.u32 {%r4,%r5,%r6,%r7}, [%rd3];
ld.global.v2.u32 {%r8,%r9}, [%rd3+16];
ld.global.v2.u32 {%r10,%r11}, [%rd3+24];
mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%r12,%r13}, {%r4,%r5,%r6,%r7}, {%r8,%r9}, {%r10,%r11};
mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%r14,%r15}, {%r4,%r5,%r6,%r7}, {%r8,%r9}, {%r12,%r13};
mul.wide.u32 %rd4, %r2, 16;
add.u64 %rd5, %rd1, %rd4;
st.global.v4.u32 [%rd5], {%r12,%r13,%r14,%r15};
ret;
}
'''


def cuda_worker(directory):
    """Runs in a fresh process so translation configuration precedes DLL load."""
    root = Path(os.environ['LOCALAPPDATA']) / 'Temp/opencode'
    paths = [os.add_dll_directory(str(p)) for p in
             (Path(r'C:\Program Files\AMD\ROCm\7.2\bin'), root / 'd4r-runtime/d4r/zluda-c10')]
    dll = root / 'd4r-runtime/d4r/zluda-c10/nvcuda.dll'
    api = ct.CDLL(str(dll))
    ptr, u64 = ct.c_void_p, ct.c_uint64
    signatures = {'cuInit':[ct.c_uint], 'cuDeviceGet':[ct.POINTER(ct.c_int), ct.c_int],
        'cuCtxCreate_v2':[ct.POINTER(ptr), ct.c_uint, ct.c_int],
        'cuModuleLoadData':[ct.POINTER(ptr), ptr],
        'cuModuleGetFunction':[ct.POINTER(ptr), ptr, ct.c_char_p],
        'cuMemAlloc_v2':[ct.POINTER(u64), ct.c_size_t],
        'cuMemcpyHtoD_v2':[u64,ptr,ct.c_size_t], 'cuMemcpyDtoH_v2':[ptr,u64,ct.c_size_t],
        'cuLaunchKernel':[ptr]+[ct.c_uint]*7+[ptr,ptr,ptr], 'cuCtxSynchronize':[],
        'cuMemFree_v2':[u64], 'cuModuleUnload':[ptr], 'cuCtxDestroy_v2':[ptr]}
    calls=[]
    def call(name,*args):
        fn=getattr(api,name);fn.argtypes=signatures[name];fn.restype=ct.c_int
        code=fn(*args);calls.append({'call':name,'result':code})
        if code: raise RuntimeError(f'{name}: {code}')
    dev=ct.c_int();ctx=ptr();module=ptr();fn=ptr();ip=u64();op=u64()
    inp=np.load(directory/'ptx-input.npy');out=np.zeros((len(inp),32,4),np.uint32)
    try:
        call('cuInit',0);call('cuDeviceGet',ct.byref(dev),0)
        call('cuCtxCreate_v2',ct.byref(ctx),0,dev)
        text=ct.create_string_buffer((directory/'probe.ptx').read_bytes()+b'\0')
        call('cuModuleLoadData',ct.byref(module),text)
        call('cuModuleGetFunction',ct.byref(fn),module,b'k32_probe_ptx')
        call('cuMemAlloc_v2',ct.byref(ip),inp.nbytes);call('cuMemAlloc_v2',ct.byref(op),out.nbytes)
        call('cuMemcpyHtoD_v2',ip,inp.ctypes.data,inp.nbytes)
        args=(ptr*2)(ct.addressof(ip),ct.addressof(op))
        call('cuLaunchKernel',fn,len(inp),1,1,32,1,1,0,None,args,None)
        call('cuCtxSynchronize');call('cuMemcpyDtoH_v2',out.ctypes.data,op,out.nbytes)
        np.save(directory/'translated-u32.npy',out)
    finally:
        if op.value: call('cuMemFree_v2',op)
        if ip.value: call('cuMemFree_v2',ip)
        if module.value: call('cuModuleUnload',module)
        if ctx.value: call('cuCtxDestroy_v2',ctx)
        (directory/'cuda-calls.json').write_text(json.dumps(calls,indent=2))
    for p in paths: p.close()


def ptx_inputs(cases):
    result=np.zeros((len(cases),32,8),np.uint32)
    def codes(values):
        # Bit-exact round trip, including signed zero. Reject non-E4M3 inputs.
        table=lo.me.E4M3.astype(np.float16).view(np.uint16)
        result = []
        for value in values.view(np.uint16):
            matches = np.flatnonzero(table == value)
            if not matches.size:
                raise ValueError('operand cannot be encoded losslessly as E4M3')
            result.append(int(matches[0]))
        return result
    def pack(v): return sum(int(x) << (8*i) for i,x in enumerate(v))
    for n,(_,a,b,c) in enumerate(cases):
        av,bv=codes(a),codes(b)
        ch=int(c.view(np.uint16));cw=ch|(ch<<16)
        for lane in range(32):
            k=4*(lane&3)
            result[n,lane]=[pack(av[k:k+4]),pack(av[k:k+4]),
                pack(av[k+16:k+20]),pack(av[k+16:k+20]),
                pack(bv[k:k+4]),pack(bv[k+16:k+20]),cw,cw]
    return result


def decode_fragments(words):
    """PTX m16n8k32 fragment ABI, preserving raw E4M3 and binary16 bits."""
    if words.shape != (32, 10) or words.dtype != np.uint32:
        raise ValueError('expected 32 lanes of ten raw uint32 fragment words')
    a = np.zeros((16, 32), np.uint8)
    b = np.zeros((32, 8), np.uint8)
    c = np.zeros((16, 8), np.uint16)
    d = c.copy()
    for lane in range(32):
        group, thread = lane >> 2, lane & 3
        av = words[lane, :4].copy().view(np.uint8)
        bv = words[lane, 4:6].copy().view(np.uint8)
        cv = words[lane, 6:8].copy().view(np.uint16)
        dv = words[lane, 8:10].copy().view(np.uint16)
        for i in range(16):
            row = group + (8 if (i // 4) % 2 else 0)
            k = 4 * thread + (i & 3) + (16 if i >= 8 else 0)
            a[row, k] = av[i]
        for i in range(8):
            b[4 * thread + (i & 3) + (16 if i >= 4 else 0), group] = bv[i]
        for i in range(4):
            row, col = group + (8 if i >= 2 else 0), 2 * thread + (i & 1)
            c[row, col], d[row, col] = cv[i], dv[i]
    return a, b, c, d


def exact_dot(a, b, c=0):
    # Every finite E4M3 is an integer multiple of 2^-9. Integer products
    # provide an oracle independent of numpy/BLAS or any GPU reduction tree.
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    finite = lo.me.E4M3[np.isfinite(lo.me.E4M3)]
    if (a.ndim != 1 or a.shape != b.shape or a.size == 0 or
            not np.all(np.isin(a, finite)) or not np.all(np.isin(b, finite)) or
            not np.isfinite(c) or float(np.float16(c)) != float(c)):
        raise ValueError('expected equal nonempty E4M3 vectors and a finite binary16 accumulator')
    ai = np.rint(a * 512).astype(np.int64)
    bi = np.rint(b * 512).astype(np.int64)
    return sum(int(x) * int(y) for x, y in zip(ai, bi)) / 262144 + float(c)


def korder(s):
    return [4 * (2 * s + (i >> 3)) + (i & 7) if (i & 7) < 4 else
            16 + 4 * (2 * s + (i >> 3)) + (i & 7) - 4 for i in range(16)]


HIP = r'''
#include <hip/hip_runtime.h>
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
extern "C" __global__ void k32_probe(const unsigned* input, unsigned* output) {
    int test = blockIdx.x, lane = threadIdx.x;
    h16 a0, a1, b0, b1;
    const unsigned* p = input + test * 65;
    int mode = blockIdx.y;
    for (int i=0; i<16; ++i) {
        int k0 = (i & 7) < 4 ? 4*(i>>3)+(i&7) : 16+4*(i>>3)+(i&7)-4;
        int k1 = k0+8;
        if (mode == 2) { k0=i; k1=i+16; }
        _Float16 scale = mode == 1 ? (_Float16)0x1p-8f : (_Float16)1;
        a0[i]=__builtin_bit_cast(_Float16,(unsigned short)p[k0])*scale;
        a1[i]=__builtin_bit_cast(_Float16,(unsigned short)p[k1])*scale;
        b0[i]=__builtin_bit_cast(_Float16,(unsigned short)p[32+k0])*scale;
        b1[i]=__builtin_bit_cast(_Float16,(unsigned short)p[32+k1])*scale;
    }
    float c0=(float)__builtin_bit_cast(_Float16,(unsigned short)p[64]);
    if(mode==1) c0*=0x1p-16f;
    f8 c=(f8){c0,c0,c0,c0,c0,c0,c0,c0};
    f8 first=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0,b0,c);
    f8 last=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1,b1,first);
    unsigned* out=output+((test*3+mode)*32+lane)*48;
    f8 rounded;
    for(int i=0;i<8;++i) {
        float h=(float)(_Float16)(last[i]*(mode==1?0x1p16f:1.0f));
        rounded[i]=h*(mode==1?0x1p-16f:1.0f);
    }
    f8 next=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0,b0,rounded);
    next=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1,b1,next);
    f8 fast=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0,b0,last);
    fast=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1,b1,fast);
    // Literal-index stores are generated below, including the actual conversion.
    STORES
}
'''


def run_hip(module, inputs, count, hip, log):
    directory = os.add_dll_directory(str(hip / 'bin'))
    api = ct.CDLL(str(hip / 'bin/amdhip64_7.dll'))
    ptr = ct.c_void_p
    signatures = {'hipInit': [ct.c_uint], 'hipSetDevice': [ct.c_int],
        'hipDeviceGetName': [ptr, ct.c_int, ct.c_int],
        'hipRuntimeGetVersion': [ct.POINTER(ct.c_int)],
        'hipMalloc': [ct.POINTER(ptr), ct.c_size_t],
        'hipMemcpy': [ptr, ptr, ct.c_size_t, ct.c_int], 'hipFree': [ptr],
        'hipModuleLoad': [ct.POINTER(ptr), ct.c_char_p],
        'hipModuleUnload': [ptr], 'hipModuleGetFunction': [ct.POINTER(ptr), ptr, ct.c_char_p],
        'hipModuleLaunchKernel': [ptr] + [ct.c_uint]*7 + [ptr, ptr, ptr],
        'hipDeviceSynchronize': []}
    calls = []
    def call(name, *args):
        fn = getattr(api, name)
        fn.argtypes = signatures[name]
        fn.restype = ct.c_int
        code = fn(*args)
        calls.append({'call': name, 'result': code})
        if code:
            raise RuntimeError(f'{name}: {code}')
    inp, out, mod, fn = ptr(), ptr(), ptr(), ptr()
    result = np.zeros((count, 3, 32, 8, 6), np.uint32)
    try:
        call('hipInit', 0)
        call('hipSetDevice', 0)
        name=ct.create_string_buffer(256)
        version=ct.c_int()
        call('hipDeviceGetName',name,256,0)
        call('hipRuntimeGetVersion',ct.byref(version))
        calls.append({'device':name.value.decode(),'runtime_version':version.value,'ordinal':0})
        call('hipMalloc', ct.byref(inp), inputs.nbytes)
        call('hipMalloc', ct.byref(out), result.nbytes)
        call('hipMemcpy', inp, inputs.ctypes.data, inputs.nbytes, 1)
        call('hipModuleLoad', ct.byref(mod), os.fsencode(module))
        call('hipModuleGetFunction', ct.byref(fn), mod, b'k32_probe')
        args = (ptr * 2)(ct.addressof(inp), ct.addressof(out))
        call('hipModuleLaunchKernel', fn, count, 3, 1, 32, 1, 1, 0, None, args, None)
        call('hipDeviceSynchronize')
        call('hipMemcpy', result.ctypes.data, out, result.nbytes, 2)
    finally:
        if mod.value: call('hipModuleUnload', mod)
        if inp.value: call('hipFree', inp)
        if out.value: call('hipFree', out)
        directory.close()
        log.write_text(json.dumps(calls, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--trace', type=Path, required=True)
    ap.add_argument('--control', type=Path, required=True)
    ap.add_argument('--experiment', type=Path, required=True)
    ap.add_argument('--original-trace', type=Path, required=True)
    a = ap.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    cases = []
    for label, trace in [('mismatch', a.trace), ('control', a.control)]:
        act = np.load(trace / 'h1-native.npy')[2 if label=='mismatch' else 34, :32]
        weight = np.load(trace / 'q-weights-native.npy')[:32, 40]
        cases.append((label, act, weight, np.float16(0)))
    z, one = np.zeros(32, np.float16), np.ones(32, np.float16)
    cases += [('zero', z, one, np.float16(0)), ('nonzero-C', z, z, np.float16(.5)),
              ('positive', one, one, np.float16(0)),
              ('negative', -one, one, np.float16(0)),
              ('cancellation', np.tile(np.array([1, -1], np.float16), 16), one, np.float16(.5)),
              ('subnormal-e4m3', one/512, one/512, np.float16(0)),
              ('maximum', one*448, one*448, np.float16(0))]
    cases.append(('mismatch-nonzero-C', cases[0][1], cases[0][2], np.float16(.125)))
    edge_a, edge_b = z.copy(), z.copy()
    edge_a[:2] = [1, 1/32]
    edge_b[:2] = [1, 1/64]
    cases += [('positive-halfway', edge_a, edge_b, np.float16(0)),
              ('negative-halfway', -edge_a, edge_b, np.float16(0))]
    clean_negative = -edge_a
    clean_negative[clean_negative == 0] = np.float16(0)
    cases.append(('negative-halfway-positive-zeros', clean_negative, edge_b, np.float16(0)))
    inputs = np.array([list(x.view(np.uint16)) + list(y.view(np.uint16)) + [c.view(np.uint16)]
                       for _, x, y, c in cases], np.uint32)
    inputs.tofile(a.out / 'input-u32.bin')
    stores = '\n'.join(f'#ifndef RESULT_ONLY\n'
        f'out[{i*6}] = __builtin_bit_cast(unsigned, first[{i}]);\n'
        f'out[{i*6+1}] = __builtin_bit_cast(unsigned, last[{i}]);\n#endif\n'
        f'out[{i*6+2}] = __builtin_bit_cast(unsigned short, (_Float16)(last[{i}]*(mode==1?0x1p16f:1.0f)));\n'
        f'out[{i*6+3}] = __builtin_bit_cast(unsigned, (float)((_Float16)(last[{i}]*(mode==1?0x1p16f:1.0f))));\n'
        f'out[{i*6+4}] = __builtin_bit_cast(unsigned short, (_Float16)(next[{i}]*(mode==1?0x1p16f:1.0f)));\n'
        f'out[{i*6+5}] = __builtin_bit_cast(unsigned short, (_Float16)(fast[{i}]*(mode==1?0x1p16f:1.0f)));'
        for i in range(8))
    src = a.out / 'probe.hip'
    src.write_text(HIP.replace('STORES', stores))
    baseline = json.loads((a.experiment / 'report.json').read_text())
    cmd = baseline['builds']['conservative']['command']
    module = a.out / 'probe.hsaco'
    cmd = cmd[:cmd.index('-o')-1] + [str(src), '-o', str(module), '--no-gpu-bundle-output']
    contract.command(cmd, a.out / 'build.log')
    hip = Path(r'C:\Program Files\AMD\ROCm\7.2')
    results = [run_hip(module, inputs, len(cases), hip, a.out / f'hip-{i}.json') for i in range(2)]
    if not np.array_equal(*results): raise RuntimeError('nondeterministic microprobe')
    result = results[0]
    np.save(a.out / 'native-u32.npy', result)
    control_module = a.out / 'result-only.hsaco'
    control_cmd = cmd.copy()
    control_cmd[control_cmd.index('-o') + 1] = str(control_module)
    control_cmd.insert(1, '-DRESULT_ONLY')
    contract.command(control_cmd, a.out / 'result-only-build.log')
    control_result = run_hip(control_module, inputs, len(cases), hip, a.out / 'result-only-hip.json')
    if not np.array_equal(control_result[..., 2:], result[..., 2:]):
        raise RuntimeError('microprobe diagnostics changed ordinary arithmetic outputs')
    # Test production HLSL conversion helpers on the same input bits.
    repo=Path(__file__).resolve().parents[1]
    dxc=Path(r'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe')
    hlsl_cmd=[dxc,'-T','cs_6_2','-E','K32Probe','-O3',repo/'src/native/k32_probe_cs.hlsl',
              '-Fo',a.out/'probe.cso']
    contract.command(hlsl_cmd,a.out/'hlsl-build.log')
    (a.out/'params.bin').write_bytes(bytes(256))
    (a.out/'graph.txt').write_text(f'''version 1
resource input {inputs.nbytes} input-u32.bin
resource output {len(cases)*16}
resource scratch 16
stage k32
shader probe.cso
threads 32 1 1
groups {len(cases)} 1 1
params params.bin 256
srv 0 input
srv 1 input
srv 2 input
srv 3 input
uav 0 output
uav 1 scratch
dump output 0 {len(cases)*16} k32.bin
''')
    for seed in (0,1):
        lf.run(str(a.out),str(repo/'build/native/Release/m_network_stage.exe'),str(a.out),seed,
               str(a.out/f'hlsl-{seed}'))
    if (a.out/'hlsl-0/k32.bin').read_bytes() != (a.out/'hlsl-1/k32.bin').read_bytes():
        raise RuntimeError('HLSL not deterministic')
    hlsl=np.fromfile(a.out/'hlsl-0/k32.bin',np.uint32).reshape(-1,4)
    # This is an operator control, NOT original enc1 runtime intermediate evidence.
    np.save(a.out/'ptx-input.npy',ptx_inputs(cases))
    (a.out/'probe.ptx').write_text(PTX)
    (a.out/'empty-native').mkdir()
    env={k:v for k,v in os.environ.items() if not k.startswith(('D4R_','ZLUDA_'))}
    settings={'D4R_ZLUDA_NATIVE_DIR':str(a.out/'empty-native'), 'D4R_ZLUDA_VERBOSE':'1',
        'D4R_ZLUDA_WMMA':'1','D4R_ZLUDA_WMMA_FP8':'1','D4R_ZLUDA_WMMA_FP8_NATIVE':'0',
        'D4R_ZLUDA_WMMA_F16_REFERENCE':'1','ZLUDA_CACHE_DIR':str(a.out/'jit-cache')}
    env.update(settings)
    cuda_cmd=[sys.executable,Path(__file__).resolve(),'--cuda-worker',a.out]
    text=contract.command(cuda_cmd,a.out/'cuda.log',env)
    first_translated = (a.out/'translated-u32.npy').read_bytes()
    contract.command(cuda_cmd,a.out/'cuda-repeat.log',env)
    if (a.out/'translated-u32.npy').read_bytes() != first_translated:
        raise RuntimeError('translated microprobe not deterministic')
    translated=np.load(a.out/'translated-u32.npy').view(np.uint16).reshape(len(cases),32,8)
    if 'backend=translated' not in text or 'backend=native' in text or 'module_direct' in text:
        raise RuntimeError('translated execution not proven; see cuda.log')
    rows = []
    original_report = json.loads((a.original_trace / 'report.json').read_text())
    if original_report['status'] != 'pass' or not all(
            v['exact'] for v in original_report['ordinary_outputs'].values()):
        raise RuntimeError('original PTX instrumentation not validated')
    if original_report.get('dispatch',{}).get('backend') != 'translated':
        raise RuntimeError('original PTX dispatch not proven')
    original_fragments = np.load(a.original_trace / 'fragments-u32.npy')
    native_cmd = baseline['runs']['conservative-first']['command']
    capture = Path(native_cmd[native_cmd.index('--fixture-dir') + 1])
    params, raw = lo.raw_layer(str(capture), 'enc1')
    sette_cmd = baseline['sette_execution']['commands'][0]
    fixture = Path(sette_cmd[sette_cmd.index('run') + 1])
    def field(data):
        return data.reshape(2, params.H, params.W, 32).transpose(1,2,0,3).reshape(params.H,params.W,64)
    reference = field(raw['output'])
    sette = field(np.fromfile(fixture/'results/out-out.bin',np.uint8))
    for idx, (name, act, weight, c) in enumerate(cases):
        expected = exact_dot(act, weight, c)
        with np.errstate(over='ignore'):
            half = int(np.float16(expected).view(np.uint16))
            cpu = np.float16(lo.me.mma_chain(act[None, :], weight[:, None], np.array([[c]]))[0, 0])
            cpu_twice = np.float16(lo.me.mma_chain(np.tile(act,2)[None,:], np.tile(weight,2)[:,None],
                                                 np.array([[c]]))[0,0])
        if int(cpu.view(np.uint16)) != half:
            raise RuntimeError('production CPU helper differs from independent integer oracle')
        row = {'name': name, 'a_f16_hex': [f'{int(v):04x}' for v in act.view(np.uint16)],
               'b_f16_hex': [f'{int(v):04x}' for v in weight.view(np.uint16)],
               'c_f16_hex': f'{int(c.view(np.uint16)):04x}', 'exact': expected,
               'exact_rne_f16_hex': f'{half:04x}', 'modes': {}}
        row['cpu_f16_hex'] = f'{int(cpu.view(np.uint16)):04x}'
        packed = ptx_inputs([(name, act, weight, c)])
        a_codes, b_codes, _, _ = decode_fragments(
            np.concatenate([packed[0], np.zeros((32, 2), np.uint32)], axis=1))
        row.update({'a_e4m3_hex':[f'{int(x):02x}' for x in a_codes[0]],
                    'b_e4m3_hex':[f'{int(x):02x}' for x in b_codes[:, 0]],
                    'a_values':act.astype(float).tolist(), 'b_values':weight.astype(float).tolist(),
                    'exact_first_k16':exact_dot(act[korder(0)], weight[korder(0)], c),
                    'exact_second_k16_product_sum':exact_dot(act[korder(1)], weight[korder(1)])})
        if idx < 2:
            # Wave 1 = natural head 1, Q columns 32..47 in pair order.
            oa, ob, oc, od = decode_fragments(original_fragments[idx, 32:64])
            if (not np.array_equal(lo.me.E4M3[oa[2]], act) or
                    not np.array_equal(lo.me.E4M3[ob[:,4]], weight) or oc[2,4] != c.view(np.uint16)):
                raise RuntimeError('original PTX operands or C differ: stop upstream')
            token = 2 if idx==0 else 34
            row['original_ptx']={'block':[4,1] if idx==0 else [2,2], 'window':0 if idx==0 else 2,
                'token':token,'natural_channel':40,'head':1,'mma_row':2,'mma_col':4,
                'd_lane':10,'d_half_in_fragment':0,'native_lane':4,'native_acc_element':1,
                'native_wave':1 if idx==0 else 3,'f16_hex':f'{int(od[2,4]):04x}',
                'operands_and_initial_accumulator_identical':True}
            bx,by=row['original_ptx']['block']
            x,y=coordinates(params,bx,by)
            sl=slice(0,16) if idx==0 else slice(32,48)
            mismatches=int((sette[y[sl],x[sl]] != reference[y[sl],x[sl]]).sum())
            if (idx == 0 and mismatches == 0) or (idx == 1 and mismatches != 0):
                raise RuntimeError('mismatch/control window classification changed')
            row['original_ptx']['field_xy']=[int(x[token]),int(y[token])]
            row['original_ptx']['window_final_output_mismatches']=mismatches
            native_half=np.load((a.trace if idx==0 else a.control)/'q-k32-first-native.npy')[token,40]
            if result[idx,0,0,0,2] != native_half.view(np.uint16):
                raise RuntimeError('microprobe does not reproduce selected original native result')
        if hlsl[idx,2] != half:
            raise RuntimeError('CPU/HLSL mismatch')
        if hlsl[idx,3] != cpu_twice.view(np.uint16):
            raise RuntimeError('CPU/HLSL successive K32 mismatch')
        row['hlsl_f16_hex']=f'{int(hlsl[idx,2]):04x}'
        row['hlsl_second_f16_hex']=f'{int(hlsl[idx,3]):04x}'
        if not np.all(translated[idx,:,:4]==translated[idx,0,0]):
            raise RuntimeError('translated replicated tile differs')
        row['translated_f16_hex']=f'{int(translated[idx,0,0]):04x}'
        row['translated_second_f16_hex']=f'{int(translated[idx,0,4]):04x}'
        if idx < 2 and row['translated_f16_hex'] != row['original_ptx']['f16_hex']:
            raise RuntimeError('isolated PTX differs from selected original runtime MMA')
        for mode, label in enumerate(('d4r-order', 'scaled-d4r-order', 'contiguous-order')):
            values = result[idx, mode]
            if not np.all(values == values[0, 0]):
                raise RuntimeError('replicated tile differs across lane/row/column')
            v = values[0, 0]
            fp = v[1].view(np.float32) * (65536 if mode==1 else 1)
            with np.errstate(over='ignore'):
                rounded = np.float16(fp).view(np.uint16)
            if int(rounded) != int(v[2]):
                raise RuntimeError('microprobe FP32/FP16 dump inconsistent')
            row['modes'][label] = {'first_f32_hex': f'{int(v[0]):08x}',
                'last_f32_hex': f'{int(v[1]):08x}', 'f16_hex': f'{int(v[2]):04x}',
                'second_f16_hex': f'{int(v[4]):04x}', 'fast_second_f16_hex': f'{int(v[5]):04x}',
                'first_f32': float(v[0].view(np.float32)), 'last_f32': float(v[1].view(np.float32))}
        rows.append(row)
    report = {'source_sha256': contract.sha(src), 'module_sha256': contract.sha(module),
        'build_command': cmd, 'input_sha256': contract.sha(a.out / 'input-u32.bin'),
        'hlsl_build_command':list(map(str,hlsl_cmd)), 'hlsl_sha256':contract.sha(a.out/'probe.cso'),
        'cuda_command':list(map(str,cuda_cmd)), 'cuda_environment':settings,
        'cuda_dll_sha256':contract.sha(Path(os.environ['LOCALAPPDATA'])/'Temp/opencode/d4r-runtime/d4r/zluda-c10/nvcuda.dll'),
        'ptx_sha256':contract.sha(a.out/'probe.ptx'), 'deterministic': True, 'cases': rows}
    report['diagnostics_preserve_microprobe_results'] = True
    report['result_only_build_command'] = control_cmd
    report['original_trace_report_sha256'] = contract.sha(a.original_trace/'report.json')
    report['original_fragments_sha256'] = contract.sha(a.original_trace/'fragments-u32.npy')
    report['baseline_report_sha256'] = contract.sha(a.experiment/'report.json')
    report['source_commit'] = baseline['source_commit']
    report['compiler_sha256'] = baseline['compiler_sha256']
    report['hip_runtime_sha256'] = contract.sha(hip/'bin/amdhip64_7.dll')
    report['sette_probe_sha256'] = contract.sha(repo/'build/native/Release/m_network_stage.exe')
    report['hlsl_sources'] = {p.name:contract.sha(p) for p in
        (repo/'src/native/k32_probe_cs.hlsl',repo/'src/native/swin_layer_cs.hlsl')}
    report['cpu_helper_sha256'] = contract.sha(lo.me.__file__)
    report['reference_trace_hashes']={str(p):contract.sha(p) for directory in (a.trace,a.control)
        for p in (directory/'report.json',directory/'h1-native.npy',directory/'q-weights-native.npy')}
    # Retain actual generated instructions, not just the source builtin spelling.
    db=sqlite3.connect('file:'+str(a.out/'jit-cache/zluda2.db')+'?mode=ro',uri=True)
    cached=db.execute('select hash,compiler_version,zluda_version,device,backend_key,binary from modules').fetchall()
    db.close()
    if len(cached)!=1: raise RuntimeError('unexpected translated module inventory')
    identity= dict(zip(('cache_hash','compiler_version','zluda_version','device','backend_key'),cached[0][:-1]))
    translated_module=a.out/'translated.hsaco'
    translated_module.write_bytes(cached[0][-1])
    identity['sha256']=contract.sha(translated_module)
    report['translated_module']=identity
    objdump=hip/'bin/llvm-objdump.exe'
    for label,artifact in [('native',module),('translated',translated_module)]:
        contract.command([objdump,'-d',artifact],a.out/f'{label}-isa.txt')
    report['isa_decoder_sha256']=contract.sha(objdump)
    report['cpu_helper_contract']='model_enc3.mma_chain, independently checked by integer E4M3 products'
    (a.out / 'report.json').write_text(json.dumps(report, indent=2))
    print('case | exact/CPU | HLSL | native | translated')
    for row in rows:
        print(f'{row["name"]} | {row["exact_rne_f16_hex"]} | {row["hlsl_f16_hex"]} | '
              f'{row["modes"]["d4r-order"]["f16_hex"]} | {row["translated_f16_hex"]}')


if __name__ == '__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--cuda-worker':
        cuda_worker(Path(sys.argv[2]))
    else:
        main()
