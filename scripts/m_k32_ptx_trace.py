#!/usr/bin/env python3
"""Replay existing enc1 PTX and dump only its first Q MMA in two CTAs.

Run in a fresh process with explicit translation settings. Reject instrumentation
if ANY ordinary output allocation differs from the existing translated capture.
No new frame capture or pinned-source modification is performed.
"""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from m_enc1_contract import sha
from m_capture_provenance import verify_translated


def instrument(text):
    pattern = (r'mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s*'
               r'\{([^}]+)\},\s*\{([^}]+)\},\s*\{([^}]+)\},\s*\{([^}]+)\};')
    match = re.search(pattern, text)
    if not match: raise ValueError('original first MMA missing')
    d, a, b, c = [[v.strip() for v in g.split(',')] for g in match.groups()]
    if list(map(len, (d,a,b,c))) != [2,4,2,2]: raise ValueError('unexpected fragments')
    regs = a+b+c+d
    stores='\n'.join(f'st.global.u32 [%trace_addr+{4*i}], {reg};' for i,reg in enumerate(regs))
    trace = '''
mov.u32 %trace_x, %ctaid.x;
mov.u32 %trace_y, %ctaid.y;
setp.eq.u32 %trace_p0, %trace_x, 4;
setp.eq.u32 %trace_p1, %trace_y, 1;
and.pred %trace_p0, %trace_p0, %trace_p1;
setp.eq.u32 %trace_p1, %trace_x, 2;
setp.eq.u32 %trace_p2, %trace_y, 2;
and.pred %trace_p1, %trace_p1, %trace_p2;
or.pred %trace_p2, %trace_p0, %trace_p1;
@!%trace_p2 bra TRACE_DONE;
mov.u32 %trace_tid, %tid.x;
mov.u32 %trace_z, %tid.z;
mad.lo.u32 %trace_tid, %trace_z, 32, %trace_tid;
selp.u32 %trace_slot, 0, 64, %trace_p0;
add.u32 %trace_tid, %trace_tid, %trace_slot;
mul.wide.u32 %trace_offset, %trace_tid, 40;
mov.u64 %trace_addr, k32_trace;
add.u64 %trace_addr, %trace_addr, %trace_offset;
''' + stores + '\nTRACE_DONE:\n'
    text = text[:match.end()] + trace + text[match.end():]
    original_sites = list(re.finditer(pattern, original_text := match.string))
    control_site = None
    if len(original_sites) > 8:
        # Four N tiles per token window: site 8 starts window 2. Only this
        # site's control CTA overwrites the window-0 control record above.
        control = list(re.finditer(pattern, text))[8]
        cd, ca, cb, cc = [[v.strip() for v in g.split(',')] for g in control.groups()]
        stores2='\n'.join(f'st.global.u32 [%trace_addr+{4*i}], {reg};'
                          for i,reg in enumerate(ca+cb+cc+cd))
        extra='''
mov.u32 %trace_x, %ctaid.x;
mov.u32 %trace_y, %ctaid.y;
setp.eq.u32 %trace_p0, %trace_x, 2;
setp.eq.u32 %trace_p1, %trace_y, 2;
and.pred %trace_p0, %trace_p0, %trace_p1;
@!%trace_p0 bra TRACE_CONTROL_DONE;
mov.u32 %trace_tid, %tid.x;
mov.u32 %trace_z, %tid.z;
mad.lo.u32 %trace_tid, %trace_z, 32, %trace_tid;
add.u32 %trace_tid, %trace_tid, 64;
mul.wide.u32 %trace_offset, %trace_tid, 40;
mov.u64 %trace_addr, k32_trace;
add.u64 %trace_addr, %trace_addr, %trace_offset;
''' + stores2 + '\nTRACE_CONTROL_DONE:\n'
        text=text[:control.end()]+extra+text[control.end():]
        control_site={'original_line':original_text[:original_sites[8].start()].count('\n')+1,
                      'registers':dict(zip(('D','A','B','C'),(cd,ca,cb,cc)))}
    entry = text.index('{', text.index('.visible .entry rrlite_enc1_4x4'))
    text = text[:entry+1] + '''
.reg .b32 %trace_x, %trace_y, %trace_tid, %trace_z, %trace_slot;
.reg .b64 %trace_addr, %trace_offset;
.reg .pred %trace_p0, %trace_p1, %trace_p2;
''' + text[entry+1:]
    text = text.replace('.visible .entry rrlite_enc1_4x4',
        '.visible .global .align 4 .u32 k32_trace[1280];\n.visible .entry rrlite_enc1_4x4',1)
    return text, {'line': text[:text.index(match.group(0))].count('\n')+1,
                  'original_line': match.string[:match.start()].count('\n')+1,
                  'control_site':control_site,
                  'registers':dict(zip(('D','A','B','C'),(d,a,b,c)))}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ptx',type=Path,required=True)
    ap.add_argument('--capture',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=ap.parse_args()
    a.out=a.out.resolve()
    if not a.worker:
        if a.out.exists():
            raise FileExistsError(f'use a new output directory: {a.out}')
        command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker']
        done=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
        if not a.out.is_dir():
            raise RuntimeError(done.stdout.decode(errors='replace'))
        (a.out/'execution.log').write_bytes(done.stdout)
        if done.returncode:
            raise RuntimeError(f'PTX trace failed: see {a.out / "execution.log"}')
        report=json.loads((a.out/'report.json').read_text())
        report['dispatch']=verify_translated(done.stdout.decode(errors='replace'),'rrlite_enc1_4x4')
        report['execution_log_sha256']=sha(a.out/'execution.log')
        report['command']=command
        (a.out/'report.json').write_text(json.dumps(report,indent=2))
        print('PASS: translated dispatch, all ordinary allocations unchanged')
        return
    a.out.mkdir(parents=True,exist_ok=False)
    original=a.ptx.read_text();text,site=instrument(original)
    (a.out/'instrumented.ptx').write_text(text)
    for key in list(os.environ):
        if key.startswith(('D4R_','ZLUDA_')): del os.environ[key]
    (a.out/'empty-native').mkdir()
    settings={'D4R_ZLUDA_NATIVE_DIR':str(a.out/'empty-native'),'D4R_ZLUDA_VERBOSE':'1',
        'D4R_ZLUDA_WMMA':'1','D4R_ZLUDA_WMMA_FP8':'1','D4R_ZLUDA_WMMA_FP8_NATIVE':'0',
        'D4R_ZLUDA_WMMA_F16_REFERENCE':'1','ZLUDA_CACHE_DIR':str(a.out/'jit-cache')}
    os.environ.update(settings)
    root=Path(os.environ['LOCALAPPDATA'])/'Temp/opencode/d4r-runtime/d4r/zluda-c10'
    paths=[os.add_dll_directory(str(p)) for p in (root,Path(r'C:\Program Files\AMD\ROCm\7.2\bin'))]
    api=ct.CDLL(str(root/'nvcuda.dll'));ptr=ct.c_void_p;u64=ct.c_uint64
    sig={'cuInit':[ct.c_uint],'cuDeviceGet':[ct.POINTER(ct.c_int),ct.c_int],
        'cuCtxCreate_v2':[ct.POINTER(ptr),ct.c_uint,ct.c_int],
        'cuModuleLoadData':[ct.POINTER(ptr),ptr],
        'cuModuleGetFunction':[ct.POINTER(ptr),ptr,ct.c_char_p],
        'cuModuleGetGlobal_v2':[ct.POINTER(u64),ct.POINTER(ct.c_size_t),ptr,ct.c_char_p],
        'cuMemAlloc_v2':[ct.POINTER(u64),ct.c_size_t],
        'cuMemcpyHtoD_v2':[u64,ptr,ct.c_size_t],'cuMemcpyDtoH_v2':[ptr,u64,ct.c_size_t],
        'cuLaunchKernel':[ptr]+[ct.c_uint]*7+[ptr,ptr,ptr],'cuCtxSynchronize':[],
        'cuMemFree_v2':[u64],'cuModuleUnload':[ptr],'cuCtxDestroy_v2':[ptr]}
    calls=[]
    def call(name,*args):
        f=getattr(api,name);f.argtypes=sig[name];f.restype=ct.c_int
        code=f(*args);calls.append({'call':name,'result':code})
        if code:raise RuntimeError(f'{name}: {code}')
    ctx=ptr();mod=ptr();fn=ptr();device=ct.c_int();allocs={}
    report={'original_ptx_sha256':sha(a.ptx),'instrumented_ptx_sha256':sha(a.out/'instrumented.ptx'),
        'cuda_dll_sha256':sha(root/'nvcuda.dll'),'environment':settings,'site':site,
        'capture':str(a.capture),'ordinary_outputs':{},'status':'running'}
    try:
        call('cuInit',0);call('cuDeviceGet',ct.byref(device),0);call('cuCtxCreate_v2',ct.byref(ctx),0,device)
        payload=ct.create_string_buffer(text.encode()+b'\0')
        call('cuModuleLoadData',ct.byref(mod),payload)
        call('cuModuleGetFunction',ct.byref(fn),mod,b'rrlite_enc1_4x4')
        manifest=(a.capture/'manifest.txt').read_text().splitlines()
        for line in manifest:
            parts=line.split()
            if parts[0]=='alloc':
                idx=int(parts[1]);data=(a.capture/f'alloc-{idx}.bin').read_bytes();p=u64()
                call('cuMemAlloc_v2',ct.byref(p),len(data));allocs[idx]=(p,len(data))
                call('cuMemcpyHtoD_v2',p,ct.create_string_buffer(data),len(data))
        params=bytearray((a.capture/'args.bin').read_bytes())
        for line in manifest:
            parts=line.split()
            if parts[0]=='pointer':
                offset,idx,delta=map(int,parts[1:]);params[offset:offset+8]=(allocs[idx][0].value+delta).to_bytes(8,'little')
        launch=list(map(int,next(l for l in manifest if l.startswith('launch ')).split()[1:]))
        packed=ct.create_string_buffer(bytes(params));args=(ptr*1)(ct.addressof(packed))
        call('cuLaunchKernel',fn,*launch[:6],launch[6],None,args,None);call('cuCtxSynchronize')
        for idx,(p,size) in allocs.items():
            data=ct.create_string_buffer(size);call('cuMemcpyDtoH_v2',data,p,size)
            (a.out/f'alloc-{idx}.bin').write_bytes(data.raw)
            expected=sha(a.capture/f'gpu-output/alloc-{idx}.bin');actual=sha(a.out/f'alloc-{idx}.bin')
            report['ordinary_outputs'][str(idx)]={'expected':expected,'actual':actual,'exact':expected==actual}
        if not all(r['exact'] for r in report['ordinary_outputs'].values()):
            raise RuntimeError('instrumented PTX changed ordinary output')
        p=u64();size=ct.c_size_t();call('cuModuleGetGlobal_v2',ct.byref(p),ct.byref(size),mod,b'k32_trace')
        if size.value != 5120:raise RuntimeError('wrong diagnostic size')
        diagnostic=np.zeros((2,64,10),np.uint32);call('cuMemcpyDtoH_v2',diagnostic.ctypes.data,p,diagnostic.nbytes)
        np.save(a.out/'fragments-u32.npy',diagnostic);report['status']='pass'
    finally:
        for p,_ in allocs.values():call('cuMemFree_v2',p)
        if mod.value:call('cuModuleUnload',mod)
        if ctx.value:call('cuCtxDestroy_v2',ctx)
        report['calls']=calls;(a.out/'report.json').write_text(json.dumps(report,indent=2))
        for p in paths:p.close()
    print(json.dumps({'status':report['status'],'site':site,'outputs':report['ordinary_outputs']},indent=2))


if __name__=='__main__':
    main()
