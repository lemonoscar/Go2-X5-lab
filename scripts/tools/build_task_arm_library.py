import json, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[2]/'source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/arm_task_data'
M=json.load(open(ROOT/'usd_geometry.json'))
Q0=np.array([0,.3,.5,0,0,0.])
LO=np.array([-1.2,.25,.2,-1.35,-1.35,-1.35]);HI=np.array([1.2,2.9,2.75,1.35,1.35,1.35])
def transform(v):
 t=np.eye(4);t[:3,:3]=R.from_quat(v['q']).as_matrix();t[:3,3]=v['p'];return t
ORIG=[transform(j['origin']) for j in M['joints']];CHILD=[np.linalg.inv(transform(j['child_origin'])) for j in M['joints']]
AXES=np.array([[int(a==j['axis']) for a in 'XYZ'] for j in M['joints']])
def fk(q):
 frames={'base':np.eye(4)}
 for i,j in enumerate(M['joints']):
  motion=np.eye(4)
  if i<6:motion[:3,:3]=R.from_rotvec(AXES[i]*q[i]).as_matrix()
  frames[j['child']]=frames[j['parent']]@ORIG[i]@motion@CHILD[i]
 return frames
def tcp(q):
 f=fk(q)['arm_link6'];return f[:3,:3]@np.array([.15757,0,0])+f[:3,3]
def obb(bounds,t):
 lo,hi=np.array(bounds);return t[:3,:3]@((hi+lo)/2)+t[:3,3],t[:3,:3],(hi-lo)/2
# SAT for enclosing oriented boxes. False positives reject a path; no mesh gaps.
def overlap(a,b,margin=.003):
 ca,ra,ha=a;cb,rb,hb=b;axes=np.r_[ra.T,rb.T,np.cross(ra.T[:,None,:],rb.T[None,:,:]).reshape(-1,3)]
 axes=axes[np.linalg.norm(axes,axis=1)>1e-8];axes/=np.linalg.norm(axes,axis=1)[:,None]
 return not np.any(abs(axes@(cb-ca)) > abs(axes@ra)@ha+abs(axes@rb)@hb+margin)
# Camera housing including 20 mm above and 30 mm forward, in base frame.
M['boxes']['camera']=[[[ .29779,-.045005,.086],[.35285,.045005,.131]]]
# Rigid neighbours and mechanically overlapping wrist housings are excluded;
# keep every other link/base/camera pair.
IGNORE={frozenset((j['parent'],j['child'])) for j in M['joints']}
IGNORE|={frozenset(p) for p in [('arm_link3','arm_link5'),('arm_link3','arm_link6'),('arm_link4','arm_link6'),('arm_link4','arm_link7'),('arm_link4','arm_link8'),('arm_link5','arm_link7'),('arm_link5','arm_link8'),('arm_link7','arm_link8'),('base','camera')]}
def collisions(q):
 f=fk(q);f['camera']=np.eye(4);boxes={k:[obb(b,f[k]) for b in v] for k,v in M['boxes'].items()};bad=[]
 names=list(boxes)
 for i,a in enumerate(names):
  for b in names[i+1:]:
   if frozenset((a,b)) in IGNORE:continue
   if any(overlap(x,y,margin=.015 if 'camera' in (a,b) else .003) for ix,x in enumerate(boxes[a]) for iy,y in enumerate(boxes[b]) if not (a=='base' and ix==0 and b=='arm_link2')):bad.append((a,b))
 # Ground assumes base >= .25 m; enforce a 20-mm floor clearance in task library.
 for name,bb in boxes.items():
  if name.startswith('arm'):
   if any(c[2]-abs(rot[2])@h < -.23 for c,rot,h in bb):bad.append((name,'floor'))
 return bad
if __name__=='__main__':print('TCP stow',tcp(Q0),'collisions',collisions(Q0))

def ik(p,seed,rotation=None):
 def residual(q):
  f=fk(q)['arm_link6'];r=[(tcp(q)-p)*10,.001*(q-seed)]
  if rotation is not None:r.append(R.from_matrix(rotation.T@f[:3,:3]).as_rotvec())
  return np.concatenate(r)
 sol=least_squares(residual,seed,bounds=(LO,HI),max_nfev=100,ftol=1e-7,xtol=1e-7,gtol=1e-7)
 if np.linalg.norm(tcp(sol.x)-p)>.002:return None
 if rotation is not None and np.linalg.norm(R.from_matrix(rotation.T@fk(sol.x)['arm_link6'][:3,:3]).as_rotvec())>.03:return None
 return sol.x

def segment(a,b,speed=.4,accel=.8,tcp_speed=.06):
 # Check the actual joint-space interpolation, not just endpoints.
 count=max(3,int(np.max(abs(b-a))/.012)+2);samples=np.linspace(a,b,count)
 if any(collisions(q) for q in samples):return None
 xyz=np.array([tcp(q) for q in samples]);deriv=np.linalg.norm(np.diff(xyz,axis=0),axis=1).max()*(count-1)
 duration=max(1.,1.875*np.max(abs(b-a))/speed,np.sqrt(5.774*np.max(abs(b-a))/accel),1.875*deriv/tcp_speed)
 return duration

def build():
 rng=np.random.default_rng(20261017);library=[];fail={}
 for stage in [0,1]:
  for kind in ['reach','lift','transfer','wrist','stow','arc']:
   accepted=0
   for attempt in range(400):
    p=np.array([rng.uniform(.39,.54 if stage==0 else .68),rng.uniform(-.10,.10) if stage==0 else rng.uniform(-.28,.28),rng.uniform(.16 if stage==0 else -.02,.29)])
    if kind=='reach':points=[p,p+np.array([.06 if stage==0 else .10,0,0]),p]
    elif kind=='lift':points=[p,p-np.array([0,0,.08 if stage==0 else .16]),p]
    elif kind=='transfer':points=[p,p+[0,0,.06],p+[0,(-1 if p[1]>0 else 1)*(.10 if stage==0 else .20),.06],p+[0,(-1 if p[1]>0 else 1)*(.10 if stage==0 else .20),0]]
    elif kind in ('wrist','stow'):points=[p]
    else:points=[p+[.035*np.sin(t),.035*(1-np.cos(t)),0] for t in np.linspace(0,np.pi/2,5)]
    if any(not (.3<=v[0]<=.75 and abs(v[1])<=.35 and -.20<=v[2]<=.45) for v in points):continue
    if stage==1 and p[2]<.12:points.insert(0,np.array([p[0],p[1],.30]))
    qs=[Q0];ok=True
    for point in points:
     q=ik(point,qs[-1])
     if q is None:ok=False;break
     qs.append(q)
    if ok and kind=='wrist':
     q=ik(points[-1],qs[-1],fk(qs[-1])['arm_link6'][:3,:3]@R.from_euler('x',.22 if stage==0 else .45).as_matrix())
     if q is None:ok=False
     else:qs.append(q)
    if not ok:continue
    qs.append(Q0);dur=[]
    for a,b in zip(qs[:-1],qs[1:]):
     d=segment(a,b)
     if d is None:ok=False;break
     dur.append(d)
    if not ok:continue
    library.append(dict(stage=stage,kind=kind,q=np.array(qs).round(8).tolist(),durations=dur,tcp=np.array([tcp(q) for q in qs]).tolist(),holds=[1.]*len(dur)))
    accepted+=1
    if accepted==4:break
   print(stage,kind,accepted,flush=True)
   fail[f'{stage}/{kind}']=accepted
 if min(fail.values())<4:raise RuntimeError(f'Insufficient checked trajectories {fail}')
 json.dump(dict(frame='base',tcp_offset=[.15757,0,0],speed=.4,acceleration=.8,tcp_speed=.06,paths=library,collision_method='conservative USD mesh bounding boxes, 0.012-rad path samples; neighbouring exclusions; runtime clearance required'),open(ROOT/'library.json','w'),indent=2)
if __name__=='__main__':build()
