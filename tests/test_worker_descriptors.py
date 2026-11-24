import errno,fcntl,os,socket,sys,tempfile,unittest
from unittest.mock import Mock,patch
from dermdetect.evaluation.worker_descriptors import snapshot,require_standard_only,normalize_pinned_entropy_cache
class DescriptorTests(unittest.TestCase):
 def test_standard_and_closed_enumeration_fd(self):
  inspect=Mock(side_effect=OSError(errno.EBADF,'closed'))
  self.assertEqual(snapshot(['0','1','2','98765'],inspect=inspect),[])
  inspect.assert_called_once_with(98765)
 def test_file_not_whitelisted_even_cloexec(self):
  with tempfile.TemporaryFile() as f:
   os.set_inheritable(f.fileno(),False)
   row=snapshot([str(f.fileno())])[0]
   self.assertEqual(row,{'fd':f.fileno(),'kind':'regular','inheritable':False})
   with self.assertRaises(PermissionError) as e:require_standard_only('after_verification',[str(f.fileno())])
   self.assertIn('after_verification',str(e.exception));self.assertNotIn('name',str(e.exception))
 def test_pipe_and_socket_rejected(self):
  r,w=os.pipe()
  try:
   self.assertEqual([x['kind'] for x in snapshot([str(r),str(w)])],['pipe','pipe'])
   with self.assertRaises(PermissionError):require_standard_only('entry',[str(r),str(w)])
  finally:os.close(r);os.close(w)
  with socket.socket() as s:
   self.assertEqual(snapshot([str(s.fileno())])[0]['kind'],'socket')
   with self.assertRaises(PermissionError):require_standard_only('entry',[str(s.fileno())])
 def test_deleted_open_file_still_rejected(self):
  fd,name=tempfile.mkstemp()
  try:
   os.unlink(name)
   with self.assertRaises(PermissionError):require_standard_only('entry',[str(fd)])
  finally:os.close(fd)
 def test_permission_errors_fail_closed(self):
  for code in [errno.EACCES,errno.EPERM,errno.EIO]:
   with self.subTest(code=code),self.assertRaises(OSError):snapshot(['3'],inspect=Mock(side_effect=OSError(code,'blocked')))
 def test_bad_names_fail_closed(self):
  for names in [['x'],['-1']]:
   with self.subTest(names=names),self.assertRaises(ValueError):snapshot(names)
if __name__=='__main__':unittest.main()

class EntropyCleanupTests(unittest.TestCase):
 def cleanup(self,names,count=1):
  with patch('sys.platform','linux'),patch('sys.version_info',(3,11,13)),patch('platform.python_implementation',return_value='CPython'):
   return normalize_pinned_entropy_cache('entry',names,task_count=lambda:count)
 def test_precise_entropy_cleanup(self):
  fd=os.open('/dev/urandom',os.O_RDONLY)
  try:
   r=self.cleanup([str(fd)]);self.assertTrue(r['entropy_cache_closed'])
   with self.assertRaises(OSError):os.fstat(fd)
   self.assertEqual(len(os.urandom(1)),1)
  finally:
   try:os.close(fd)
   except OSError:pass
 def test_unknown_not_closed(self):
  with tempfile.TemporaryFile() as f:
   with self.assertRaises(PermissionError):self.cleanup([str(f.fileno())])
   os.fstat(f.fileno())
 def test_extra_unknown_prevents_all_cleanup(self):
  fd=os.open('/dev/urandom',os.O_RDONLY)
  try:
   with tempfile.TemporaryFile() as f:
    with self.assertRaises(PermissionError):self.cleanup([str(fd),str(f.fileno())])
    os.fstat(fd);os.fstat(f.fileno())
  finally:os.close(fd)
 def test_inheritable_refused(self):
  fd=os.open('/dev/urandom',os.O_RDONLY)
  try:
   os.set_inheritable(fd,True)
   with self.assertRaises(PermissionError):self.cleanup([str(fd)])
   os.fstat(fd)
  finally:os.close(fd)
 def test_multithread_refused(self):
  fd=os.open('/dev/urandom',os.O_RDONLY)
  try:
   with self.assertRaises(PermissionError):self.cleanup([str(fd)],2)
   os.fstat(fd)
  finally:os.close(fd)
 def test_no_descriptors_unchanged(self):self.assertFalse(self.cleanup([])['entropy_cache_closed'])
