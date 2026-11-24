"""Fail-closed worker FD audit; metadata only, never paths or descriptor contents."""
import errno
import json
import os
import stat

def snapshot(names=None, *, inspect=os.fstat, inheritable=os.get_inheritable):
    names=os.listdir('/proc/self/fd') if names is None else names
    records=[]
    for name in sorted(names,key=int):
        fd=int(name)
        if fd<0:raise ValueError('Negative descriptor')
        if fd<=2:continue
        try:
            mode=inspect(fd).st_mode
            flag=inheritable(fd)
        except OSError as exc:
            # Directory enumeration can name its own already-closed descriptor.
            # EBADF alone means no live descriptor; all other errors fail closed.
            if exc.errno==errno.EBADF:continue
            raise
        kind=('regular' if stat.S_ISREG(mode) else 'directory' if stat.S_ISDIR(mode)
              else 'socket' if stat.S_ISSOCK(mode) else 'pipe' if stat.S_ISFIFO(mode)
              else 'character' if stat.S_ISCHR(mode) else 'other')
        records.append({'fd':fd,'kind':kind,'inheritable':flag})
    return records

def require_standard_only(stage, names=None, **kwargs):
    records=snapshot(names,**kwargs)
    if records:
        raise PermissionError('Unexpected nonstandard descriptors: '+json.dumps({'stage':stage,'descriptors':records},sort_keys=True))
    return {'stage':stage,'nonstandard_descriptors':[]}

def normalize_pinned_entropy_cache(stage, names=None, *, task_count=None):
    """Close only the proven pinned-CPython entropy cache, then reject all extras.

    CPython3.11.13 bootstrap_hash.c verifies cached dev/inode before reuse and
    safely reopens a closed/replaced /dev/urandom handle. This is cleanup, not a
    descriptor whitelist. No unknown descriptor is closed or accepted.
    """
    import fcntl
    import platform
    import sys
    from pathlib import Path
    records=snapshot(names)
    if not records:return {'stage':stage,'entropy_cache_closed':False,'nonstandard_descriptors':[]}
    if (sys.platform!='linux' or sys.version_info[:3]!=(3,11,13)
            or platform.python_implementation()!='CPython' or hasattr(os,'getrandom')):
        raise PermissionError('Unrecognized runtime descriptor state')
    if len(records)!=1:raise PermissionError('Unknown or multiple descriptors; no cleanup performed')
    count=task_count() if task_count else len(list(Path('/proc/self/task').iterdir()))
    if count!=1:raise PermissionError('Single-threaded descriptor cleanup required')
    fd=records[0]['fd'];expected=os.stat('/dev/urandom')
    identity=lambda s:(stat.S_IFMT(s.st_mode),s.st_dev,s.st_ino,s.st_rdev)
    def verified():
        current=os.fstat(fd)
        return (stat.S_ISCHR(current.st_mode) and identity(current)==identity(expected)
                and fcntl.fcntl(fd,fcntl.F_GETFL)&os.O_ACCMODE==os.O_RDONLY
                and fcntl.fcntl(fd,fcntl.F_GETFD)&fcntl.FD_CLOEXEC
                and not os.get_inheritable(fd))
    if not verified():raise PermissionError('Unexpected descriptor; no cleanup performed')
    # Immediate identity recheck in the already single-threaded trusted worker.
    if not verified():raise PermissionError('Descriptor changed during verification')
    os.close(fd)
    require_standard_only(stage,names)
    return {'stage':stage,'entropy_cache_closed':True,'closed_fd':fd,
            'verified_device':'/dev/urandom','nonstandard_descriptors':[]}
