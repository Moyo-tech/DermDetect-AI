import io,json,math,struct,tempfile,unittest,zipfile
from pathlib import Path
from dermdetect.evaluation import metrics as m, engine as e, arrays as a

def npy(values,shape,dtype='<f8'):
 h=repr(dict(descr=dtype,fortran_order=False,shape=shape)).encode();h+=b' ' *((64-(10+len(h)+1)%64)%64)+b'\n'
 return b'\x93NUMPY\x01\x00'+struct.pack('<H',len(h))+h+struct.pack('<'+str(len(values))+a.TYPES[dtype][0],*values)

def fixture():
 rows=[dict(image_id='i'+str(i),physical_id='p'+str(i),cluster_id='c'+str(i),label=c,partition='synthetic',sha256=e.digest(c)) for i,c in enumerate(e.CLASSES)]
 images={};specs={}
 for s in e.SYSTEMS:
  k=7 if s.endswith('7') else 2;keep=rows if k==7 else [r for r in rows if e.BINARY[r['label']] is not None]
  images[s]=dict(image_ids=[r['image_id'] for r in keep],logits=[[4. if j==(e.CLASSES.index(r['label']) if k==7 else e.BINARY[r['label']]) else 0. for j in range(k)] for r in keep])
  images[s]['image_SHAs']=[r['sha256'] for r in keep];images[s]['payload_SHA']=e.digest({'image_ids':images[s]['image_ids'],'logits':images[s]['logits']})
  specs[s]=dict(classes=list(range(k)),confidence_tie_policy='include_all_ties',artifact_SHA='synthetic',temperature=1.,cutoffs=[.5,.8],malignant_probability=.5,binary_equality='malignant')
 return rows,images,specs

class Metrics(unittest.TestCase):
 def test_perfect(self):
  s=m.scores([[5,0],[0,5]],[0,1]);self.assertEqual(s['macro_F1']['value'],1);self.assertEqual(s['MCC']['value'],1);self.assertEqual(s['AUROC']['value'],1);self.assertEqual(s['AP']['value'],1)
 def test_binary_tie_and_brier(self):
  s=m.scores([[0,0],[0,0]],[0,1],cutoffs=[.5,.6]);self.assertEqual(s['confusion_matrix'],[[0,1],[0,1]]);self.assertEqual(s['binary_Brier']['value'],.25);self.assertEqual(s['multiclass_Brier']['value'],.5);self.assertEqual(s['selective'][0]['coverage'],1);self.assertIsNone(s['selective'][1]['selective_error']['value'])
 def test_ranking_ties(self):
  auc,ap=m.binary_ranking([1,0,1,0],[.9,.9,.2,.1],[1]*4);self.assertEqual(auc['value'],.625);self.assertAlmostEqual(ap['value'],7/12)
 def test_ranking_all_positive_negative(self):
  self.assertIsNone(m.binary_ranking([0],[.5],[1])[1]['value']);self.assertEqual(m.binary_ranking([1],[.5],[1])[1]['value'],1);self.assertIsNone(m.binary_ranking([1],[.5],[1])[0]['value'])
 def test_tied_aurc(self):
  s=m.scores([[0,0],[0,0]],[0,1]);self.assertEqual(s['AURC']['value'],.5);self.assertEqual(len(s['risk_coverage']),1)
 def test_missing_class(self):
  s=m.scores([[1,0]],[0]);self.assertIsNone(s['macro_F1']['value']);self.assertIsNone(s['macro_AUROC']['value']);self.assertIsNone(s['MCC']['value']);self.assertIsNone(s['balanced_accuracy']['value'])
 def test_extreme_stability(self):
  s=m.scores([[1000,-1000],[-1000,1000]],[0,1]);self.assertEqual(s['NLL']['value'],0);self.assertEqual(s['ECE_bins'][-1]['count'],2)
 def test_invalid(self):
  for z,y,t,w in [([[math.nan,0]],[0],1,None),([[1,0]],[True],1,None),([[1,0]],[0],0,None),([[1,0]],[0],1,[0]),([[1,0]],[0],1,[.5])]:
   with self.subTest(z=z,y=y,t=t,w=w),self.assertRaises(ValueError):m.scores(z,y,t,w)
 def test_integer_weights_equal_expansion(self):
  x=[[2,0],[0,1]];y=[0,1];s=m.scores(x,y,weights=[2,1]);t=m.scores([x[0],x[0],x[1]],[0,0,1]);self.assertEqual(s,t)

class Engine(unittest.TestCase):
 def test_four_systems_and_binary_exclusion(self):
  r,x,s=fixture();v=e.evaluate(r,x,s,partition='synthetic',bootstrap=True);self.assertEqual(v['systems']['med7']['physical_units'],7);self.assertEqual(v['systems']['dense2']['physical_units'],6);self.assertEqual(v['paired_intervals']['7']['replicates'],1000);self.assertEqual(v['paired_intervals']['7']['intervals']['macro_F1']['b_minus_a'],[0,0])
 def test_actual_heldout_gate_never_accepts_dict(self):
  for partition in ['internal_test','DDI','train']:
   with self.subTest(partition=partition),self.assertRaises(PermissionError):e.authority(partition,dict(status='approved_locked',systems=list(e.SYSTEMS),approval_reference='invented',evaluator_SHA='x',system_manifest_SHA='x'),dict(held_out_access=True))
 def test_payload_and_image_hash_mismatch(self):
  for field in ('payload_SHA','image_SHAs'):
   r,x,s=fixture();x['med7'][field]='wrong'
   with self.subTest(field=field),self.assertRaises(ValueError):e.evaluate(r,x,s,partition='synthetic')
 def test_order_and_cluster(self):
  r,x,s=fixture()
  with self.assertRaises(ValueError):e.aggregate(r,list(reversed(x['med7']['image_ids'])),x['med7']['logits'],7)
  q=dict(r[0],image_id='duplicate',cluster_id='other')
  with self.assertRaises(ValueError):e.aggregate(r+[q],x['med7']['image_ids']+['duplicate'],x['med7']['logits']+[x['med7']['logits'][0]],7)
 def test_mean_logits_physical_not_cluster(self):
  r,_,_=fixture();q=dict(r[0],image_id='another');ids,z,y,c=e.aggregate([r[0],q,r[1]],['i0','another','i1'],[[2]*7,[4]*7,[8]*7],7);self.assertEqual(z,((3.,)*7,(8.,)*7))
 def test_paired_variable_population(self):
  draws=list(e.paired_draws(['a','b','c','d'],[0,0,0,1],['g','g','h','i'],2));self.assertEqual(len(draws),1000);self.assertTrue(all(w[0]==w[1] for w in draws));self.assertGreater(len(set(sum(w) for w in draws)),1);self.assertEqual(draws,list(e.paired_draws(['a','b','c','d'],[0,0,0,1],['g','g','h','i'],2)))
 def test_mixed_cluster_and_wrong_seed(self):
  for labels,clusters,seed in [([0,1],['g','g'],e.SEED),([0,1],['a','b'],1)]:
   with self.assertRaises(ValueError):list(e.paired_draws(['a','b'],labels,clusters,2,seed=seed))
 def test_percentiles_no_retry(self):
  self.assertEqual(e.interval(list(range(1000))),[24.975,974.025])
  with self.assertRaises(ValueError):e.interval([0]*999+[math.nan])
 def test_record_before_review_and_no_overwrite(self):
  r,x,s=fixture();v=e.evaluate(r,x,s,partition='synthetic')
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'record.json';sha=e.record_once(p,v,input_SHA='invented',source_SHA='invented');self.assertTrue(e.permit_error_review(p,sha))
   with self.assertRaises(FileExistsError):e.record_once(p,v,input_SHA='invented',source_SHA='invented')
   with self.assertRaises(PermissionError):e.permit_error_review(p,'wrong')
 def test_temperature_changes_confidence_not_rank(self):
  self.assertGreater(m.scores([[4,0],[0,4]],[0,1],1)['ECE']['value']*-1,m.scores([[4,0],[0,4]],[0,1],2)['ECE']['value']*-1)

class Arrays(unittest.TestCase):
 def test_numeric_only(self):
  self.assertEqual(a.read(io.BytesIO(npy([1,2,3,4],(2,2))))[1],((1,2),(3,4)))
  with self.assertRaises(ValueError):a.read(io.BytesIO(npy([math.nan],(1,))))
 def test_exact_length_and_memory(self):
  for data,limit in [(npy([1],(1,))+b'x',100),(npy([1],(1,))[:-1],100),(npy([1],(1,)),1)]:
   with self.assertRaises(ValueError):a.read(io.BytesIO(data),limit)
 def test_probe_and_feature_stream(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'probe.npz'
   with zipfile.ZipFile(p,'w') as z:
    for name,values,shape in [('mean',[1,2],(2,)),('scale',[2,1],(2,)),('variance',[4,1],(2,)),('coefficients',[2,3],(1,2)),('intercept',[1],(1,)),('classes',[0,1],(2,))]:z.writestr(name+'.npy',npy(values,shape,'<i8' if name=='classes' else '<f8'))
   state=a.portable_probe(p);self.assertEqual(a.probe_logits(state,[3,4]),(0,9))
   f=Path(t)/'feature.npy';f.write_bytes(npy([1,2,3,4],(2,2),'<f4'));self.assertEqual(list(a.feature_rows(f,(2,2),[1])),[(3,4)])
   with self.assertRaises(ValueError):list(a.feature_rows(f,(3,2),[0]))
 def test_float32_transform_rounds_each_operation(self):
  state=dict(mean=(.13,),scale=(.31,),classes=(0,1),coefficients=((1.,),),intercept=(0.,))
  def f(x):return struct.unpack('<f',struct.pack('<f',x))[0]
  self.assertEqual(a.probe_logits(state,[1.]),(0.,f(f(1.-.13)/.31)))
  self.assertNotEqual(a.probe_logits(state,[1.]),a.probe_logits(state,[1.],feature_dtype='<f8'))
 def test_extra_archive_member_refused(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'probe.npz'
   with zipfile.ZipFile(p,'w') as z:z.writestr('evil.npy',b'x')
   with self.assertRaises(ValueError):a.portable_probe(p)
