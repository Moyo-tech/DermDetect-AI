"""Nonactive candidate. Exact AF_UNIXSEQPACKET_CLOEXEC exception only.
ProfileB connect permission is the proposed run profile; requires explicit scope/risk approval.
No socketpair allowance; no endpoint, broker-egress or hostile-code isolation claim.
"""
import ctypes,errno
CMP_NE=1;CMP_MASKED_EQ=7;AF_UNIX=1;SEQPACKET_CLOEXEC=0x80005
PROFILES={'socket_only','socket_and_connect'}
def rules_spec(profile):
 if profile not in PROFILES:raise ValueError('Unapproved filter profile')
 out=[('socket',errno.EPERM,[(0,CMP_NE,AF_UNIX,0)]),('socket',errno.EPERM,[(1,CMP_NE,SEQPACKET_CLOEXEC,0)])]
 out += [(name,errno.EPERM,[]) for name in ['socketpair','execve','execveat','fork','vfork']]
 if profile=='socket_only':out.append(('connect',errno.EPERM,[]))
 out += [('clone3',errno.ENOSYS,[]),('clone',errno.EPERM,[(0,CMP_MASKED_EQ,0x10000,0)])]
 return out

def install_filter(profile):
 class Cmp(ctypes.Structure):
  _fields_=[('arg',ctypes.c_uint),('op',ctypes.c_uint),('a',ctypes.c_uint64),('b',ctypes.c_uint64)]
 s=ctypes.CDLL('/lib/x86_64-linux-gnu/libseccomp.so.2')
 s.seccomp_init.argtypes=[ctypes.c_uint32];s.seccomp_init.restype=ctypes.c_void_p
 s.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];s.seccomp_syscall_resolve_name.restype=ctypes.c_int
 s.seccomp_rule_add_array.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint,ctypes.POINTER(Cmp)]
 s.seccomp_attr_set.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_uint32]
 s.seccomp_load.argtypes=[ctypes.c_void_p];s.seccomp_release.argtypes=[ctypes.c_void_p]
 ctx=s.seccomp_init(0x7fff0000)
 if not ctx:raise RuntimeError('Cannot initialize candidate filter')
 try:
  if s.seccomp_attr_set(ctx,4,1):raise RuntimeError('Cannot synchronize candidate filter')
  for name,error,comparisons in rules_spec(profile):
   nr=s.seccomp_syscall_resolve_name(name.encode())
   if nr<0:
    if name=='clone3':continue
    raise RuntimeError('Required syscall unavailable')
   arr=(Cmp*len(comparisons))(*(Cmp(*x) for x in comparisons)) if comparisons else None
   if s.seccomp_rule_add_array(ctx,0x50000|error,nr,len(comparisons),arr):raise RuntimeError('Cannot install candidate rule')
  if s.seccomp_load(ctx):raise RuntimeError('Cannot load candidate filter')
 finally:s.seccomp_release(ctx)
 return {'profile':profile,'TSYNC':True,'socket_exception':'AF_UNIX type0x80005 only','socketpair_denied':True,'connect_denied':profile=='socket_only','endpoint_restricted':False,'indirect_egress_excluded':False,'exec_fork_nonthread_process_denied':True}
