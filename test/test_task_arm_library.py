"""Numerical acceptance for the task library, independent of Isaac runtime."""
import importlib.util,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('builder',root/'scripts/tools/build_task_arm_library.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
data=json.loads((m.ROOT/'library.json').read_text());maxv=maxa=maxtcp=0.
for path in data['paths']:
 q=np.array(path['q']);assert np.all(q>=m.LO) and np.all(q<=m.HI)
 np.testing.assert_allclose(q[0],m.Q0);np.testing.assert_allclose(q[-1],m.Q0)
 for a,b,d in zip(q[:-1],q[1:],path['durations']):
  t=np.linspace(0,1,int(d/.02)+2);dt=d/(len(t)-1);blend=10*t**3-15*t**4+6*t**5
  curve=a+blend[:,None]*(b-a);v=np.diff(curve,axis=0)/dt;acc=np.diff(v,axis=0)/dt
  xyz=np.array([m.tcp(p) for p in curve]);maxv=max(maxv,abs(v).max());maxa=max(maxa,abs(acc).max());maxtcp=max(maxtcp,np.linalg.norm(np.diff(xyz,axis=0)/dt,axis=1).max())
  assert all(not m.collisions(p) for p in curve[::10])
assert maxv<=.4001 and maxa<=.8001 and maxtcp<=.0601,(maxv,maxa,maxtcp)
print('PASS',len(data['paths']),'paths; max joint speed/acceleration/TCP speed',maxv,maxa,maxtcp)
