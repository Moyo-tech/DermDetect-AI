"""Fail-closed Linux file policy plus the unchanged accepted local-IPC filter.

Landlock ABI>=3 is required: no weaker filesystem fallback. Mac tests exercise
policy/syscall contracts only, never claim Linux kernel enforcement. The narrow
AF_UNIX SOCK_SEQPACKET|CLOEXEC/connect exception is the historical trusted-worker
contract; it does not establish endpoint isolation or exclude broker egress.
"""
import ctypes
import os
from pathlib import Path
import platform
import sys

FS_EXEC=1;FS_WRITE=2;FS_READ=4;FS_READ_DIR=8
FS_REMOVE_DIR=16;FS_REMOVE_FILE=32;FS_MAKE_DIR=128;FS_MAKE_REG=256;FS_TRUNCATE=16384
HANDLED_FS=(1<<15)-1
OUTPUT_FS=FS_WRITE|FS_READ|FS_READ_DIR|FS_REMOVE_DIR|FS_REMOVE_FILE|FS_MAKE_DIR|FS_MAKE_REG|FS_TRUNCATE
LOCAL_IPC_PROFILE='socket_and_connect'

class Ruleset(ctypes.Structure):
    _fields_=[('handled_access_fs',ctypes.c_uint64)]
class PathRule(ctypes.Structure):
    _pack_=1
    _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]

class Kernel:
    def __init__(self):
        if sys.platform!='linux' or platform.machine()!='x86_64':
            raise PermissionError('Actual Linux x86_64 readiness required; no local emulation claim')
        self.libc=ctypes.CDLL(None,use_errno=True);self.libc.syscall.restype=ctypes.c_long
        self.libc.prctl.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong]
        self.libc.prctl.restype=ctypes.c_int
    def call(self,n,*args):
        result=self.libc.syscall(ctypes.c_long(n),*args)
        if result<0:raise OSError(ctypes.get_errno(),'Landlock syscall refused')
        return result
    def abi(self):return self.call(444,None,ctypes.c_size_t(0),ctypes.c_uint(1))
    def create(self,handled):
        attr=Ruleset(handled);return self.call(444,ctypes.byref(attr),ctypes.c_size_t(ctypes.sizeof(attr)),ctypes.c_uint(0))
    def add(self,ruleset,path,access):
        fd=os.open(path,os.O_PATH|os.O_CLOEXEC)
        try:
            attr=PathRule(access,fd);self.call(445,ctypes.c_int(ruleset),ctypes.c_int(1),ctypes.byref(attr),ctypes.c_uint(0))
        finally:os.close(fd)
    def restrict(self,ruleset):
        if self.libc.prctl(38,1,0,0,0):raise OSError(ctypes.get_errno(),'no_new_privs refused')
        self.call(446,ctypes.c_int(ruleset),ctypes.c_uint(0))
    def close(self,fd):os.close(fd)


def policy(*,files,read_trees,output,directories=(),devices=()):
    """No implicit '/usr' or '/teamspace' grants; explicit runtime trees only."""
    output=Path(output).resolve(strict=True)
    file_paths={Path(p).resolve(strict=True) for p in files}
    trees={Path(p).resolve(strict=True) for p in read_trees}
    dirs={Path(p).resolve(strict=True) for p in directories}
    if output==Path('/') or any(p==Path('/') for p in trees|dirs):
        raise PermissionError('Root read/write grant forbidden')
    for p in file_paths|trees|dirs:
        if p==output or p in output.parents or output in p.parents:
            raise PermissionError('Source/runtime/output overlap')
    if any(p.is_dir() for p in file_paths) or any(not p.is_dir() for p in trees|dirs):
        raise PermissionError('File/tree policy type mismatch')
    grants={}
    def add(p,bits):grants[str(p)]=grants.get(str(p),0)|bits
    for p in file_paths:add(p,FS_READ)
    for p in trees:add(p,FS_READ|FS_READ_DIR)
    for p in dirs:add(p,FS_READ_DIR)
    for p,bits in devices:
        q=Path(p).resolve(strict=True)
        if str(q) not in ('/dev/null','/dev/random','/dev/urandom'):raise PermissionError('Unapproved special device')
        add(q,bits)
    add(output,OUTPUT_FS)
    return {'minimum_abi':3,'handled_access_fs':HANDLED_FS,'grants':sorted(grants.items()),
            'local_IPC_profile':LOCAL_IPC_PROFILE,'network_exec_fork_filter_unchanged':True}


def enforce(spec,*,kernel=None):
    # Injection is solely a unit-test interface; production entry never supplies it.
    kernel=kernel or Kernel()
    if spec.get('minimum_abi')!=3 or spec.get('handled_access_fs')!=HANDLED_FS or spec.get('local_IPC_profile')!=LOCAL_IPC_PROFILE:
        raise PermissionError('Isolation contract substitution forbidden')
    abi=kernel.abi()
    if abi<3:raise PermissionError('Landlock ABI>=3 required; no ambient-read fallback')
    fd=kernel.create(HANDLED_FS)
    try:
        for path,bits in spec['grants']:kernel.add(fd,path,bits)
        kernel.restrict(fd)
    finally:kernel.close(fd)
    return {'Landlock_ABI':abi,'filesystem_enforced':True,'handled_access_fs':HANDLED_FS}


def install_seccomp():
    if sys.platform!='linux' or platform.machine()!='x86_64':raise PermissionError('Linux x86_64 only')
    from .linux_seccomp import install_filter
    return install_filter(LOCAL_IPC_PROFILE)
