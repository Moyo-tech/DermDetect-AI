"""Bounded portable NPY/NPZ readers; never import NumPy or deserialize pickle."""
import ast
import io
import math
import struct
import zipfile
from pathlib import Path

TYPES={'<f4':('f',4),'<f8':('d',8),'<i8':('q',8),'<i4':('i',4)}
def header(f):
    if f.read(6)!=b'\x93NUMPY':raise ValueError('NPY magic required')
    v=f.read(2)
    if v not in (b'\x01\x00',b'\x02\x00',b'\x03\x00'):raise ValueError('Supported NPY version required')
    n=2 if v[0]==1 else 4;b=f.read(n)
    if len(b)!=n:raise ValueError('Truncated NPY header')
    length=struct.unpack('<H' if n==2 else '<I',b)[0]
    if length>65536:raise ValueError('Bounded NPY header required')
    data=f.read(length)
    if len(data)!=length:raise ValueError('Truncated NPY header')
    m=ast.literal_eval(data.decode('utf8' if v[0]==3 else 'latin1'))
    if set(m)!=set(['descr','fortran_order','shape']) or m['descr'] not in TYPES or m['fortran_order'] is not False or not isinstance(m['shape'],tuple) or not 1<=len(m['shape'])<=2 or any(type(n) is not int or n<1 for n in m['shape']):raise ValueError('Finite rectangular numeric C-order arrays only; no objects/pickle')
    return m

def read(f,max_bytes=2**21):
    m=header(f);kind,size=TYPES[m['descr']];count=math.prod(m['shape']);n=count*size
    if n>max_bytes:raise ValueError('Portable array exceeds bounded memory')
    raw=f.read(n)
    if len(raw)!=n or f.read(1):raise ValueError('Exact NPY payload length required')
    values=struct.unpack('<'+str(count)+kind,raw)
    if any(not math.isfinite(v) for v in values):raise ValueError('Nonfinite portable array')
    if len(m['shape'])==2:values=tuple(values[i:i+m['shape'][1]] for i in range(0,count,m['shape'][1]))
    return m,values

def load(path,max_bytes=2**21):
    p=Path(path)
    if p.is_symlink():raise ValueError('No array symlinks')
    with p.open('rb') as f:return read(f,max_bytes)

def portable_probe(path):
    required={'mean','scale','variance','coefficients','intercept','classes'}
    if Path(path).is_symlink():raise ValueError('No probe symlinks')
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if set(names)!={n+'.npy' for n in required} or len(names)!=len(required) or sum(i.file_size for i in z.infolist())>2**21:raise ValueError('Exact bounded numeric probe archive required')
        out={}
        for n in required:
            with z.open(n+'.npy') as f:out[n]=read(f)[1]
    d=len(out['mean']);classes=out['classes'];k=len(classes);co=out['coefficients']
    if classes!=tuple(range(k)) or k not in (2,7) or len(out['scale'])!=d or len(out['variance'])!=d or any(s<=0 for s in out['scale']) or any(v<0 for v in out['variance']) or len(co)!=(1 if k==2 else k) or any(len(r)!=d for r in co) or len(out['intercept'])!=len(co):raise ValueError('Canonical probe/scaler shapes required')
    return out

def probe_logits(state,row,*,feature_dtype='<f4'):
    if len(row)!=len(state['mean']) or any(not math.isfinite(v) for v in row):raise ValueError('Finite aligned feature row required')
    if feature_dtype not in ('<f4','<f8'):raise ValueError('Explicit float32/float64 feature arithmetic required')
    # sklearn StandardScaler.transform preserves input dtype: each in-place
    # subtraction and division rounds back to float32 for the pinned caches.
    # Fit/scaler arrays remain float64. Match these operations, never tolerances.
    def round_feature(x):return struct.unpack('<f',struct.pack('<f',x))[0] if feature_dtype=='<f4' else x
    z=tuple(round_feature(round_feature(x-m)/s) for x,m,s in zip(row,state['mean'],state['scale']))
    score=tuple(math.fsum(a*b for a,b in zip(z,co))+bias for co,bias in zip(state['coefficients'],state['intercept']))
    if any(not math.isfinite(v) for v in score):raise ValueError('Portable prediction overflow')
    return (0.,score[0]) if len(state['classes'])==2 else score

def feature_rows(path,expected_shape,selected):
    """Seek only named development rows of a hash-verified cache, one row at a time."""
    p=Path(path)
    if p.is_symlink():raise ValueError('No cache symlinks')
    with p.open('rb') as f:
        m=header(f)
        if m['shape']!=tuple(expected_shape) or m['descr']!='<f4':raise ValueError('Exact float32 cache shape required')
        offset=f.tell();rows,cols=m['shape'];n=rows*cols*4
        if p.stat().st_size!=offset+n:raise ValueError('Exact feature payload size required')
        for i in selected:
            if type(i) is not int or not 0<=i<rows:raise ValueError('Exact selected development index required')
            f.seek(offset+i*cols*4);raw=f.read(cols*4)
            if len(raw)!=cols*4:raise ValueError('Truncated feature row')
            row=struct.unpack('<'+str(cols)+'f',raw)
            if any(not math.isfinite(v) for v in row):raise ValueError('Nonfinite feature row')
            yield row
