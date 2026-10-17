"""Checked finite task trajectories; floating base, unchanged 483/12 policy ABI."""
import json
from pathlib import Path
import torch
from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg
from .unified_carry import UnifiedArmCommand
from .utils import is_env_assigned_to_terrain
from .unified_payload import randomize_tcp_payload

DATA = Path(__file__).with_name('arm_task_data')

class TaskArmCommand(UnifiedArmCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if self.robot.is_fixed_base:
            raise RuntimeError('Task-arm stabilization requires a floating base')
        for name, actuator in self.robot.actuators.items():
            if name.startswith('arm_'):
                limit=15. if name in ('arm_joint12','arm_joint3') else 3.
                if (actuator.effort_limit_sim>limit+1e-5).any() or (actuator.velocity_limit_sim>3.00001).any():
                    raise RuntimeError('Unexpected physics actuator limits: '+name)
        effort=self.robot.root_physx_view.get_dof_max_forces()[0,self.ids]
        velocity=self.robot.root_physx_view.get_dof_max_velocities()[0,self.ids]
        expected=effort.new_tensor([15.,15.,15.,3.,3.,3.])
        if not torch.allclose(effort,expected) or not torch.allclose(velocity,velocity.new_full((6,),3.)):
            raise RuntimeError('PhysX limit readback differs from task contract')
        print('TASK_ARM_CONTRACT floating base; PhysX effort',effort.tolist(),'velocity',velocity.tolist(),flush=True)
        paths = [p for p in json.loads((DATA/'library.json').read_text())['paths'] if p['stage']==cfg.stage]
        weights={'reach':.25,'lift':.25,'transfer':.15,'wrist':.15,'stow':.10,'arc':.10}
        # Offline trajectory sampling at the actual policy period. Each segment is
        # rest-to-rest quintic; duration satisfies joint speed/acceleration and TCP speed.
        curves=[]
        for p in paths:
            q=torch.tensor(p['q'],device=self.device)
            curve=[q[0:1].repeat(round(1./env.step_dt),1)]
            for i,duration in enumerate(p['durations']):
                n=int(duration/env.step_dt)+1
                t=torch.arange(1,n+1,device=self.device)/n
                blend=10*t**3-15*t**4+6*t**5
                curve.append(q[i]+blend[:,None]*(q[i+1]-q[i]))
                curve.append(q[i+1:i+2].repeat(round(cfg.hold_seconds/env.step_dt),1))
            curves.append(torch.cat(curve))
        self.lengths=torch.tensor([len(c) for c in curves],device=self.device)
        self.curves=self.target.new_zeros(len(curves),int(self.lengths.max()),6)
        for i,c in enumerate(curves):
            self.curves[i,:len(c)]=c;self.curves[i,len(c):]=c[-1]
        limits=self.robot.data.soft_joint_pos_limits[0,self.ids]
        if not ((self.curves>=limits[:,0]).all() and (self.curves<=limits[:,1]).all()):
            raise RuntimeError('Task library exceeds actual articulation soft limits')
        self.weights=torch.tensor([weights[p['kind']] for p in paths],device=self.device)
        self.path_id=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.sample=torch.zeros_like(self.path_id)
        self.running=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.metrics['task_progress']=self.clock.clone()
        self.metrics['tracking_error']=self.clock.clone()

    def _resample_command(self, env_ids):
        # Reset-only selection; never replace a target while moving.
        ids=torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
        self.path_id[ids]=torch.multinomial(self.weights,len(ids),replacement=True)
        self.sample[ids]=0;self.running[ids]=False

    def _update_command(self):
        base=self._env.command_manager.get_term('base_velocity')
        standing=(base.requested==0).all(-1)
        flat=is_env_assigned_to_terrain(self._env,'flat')
        start=standing & flat & ~self.running
        self.running|=start
        self.target.copy_(self.curves[self.path_id,self.sample])
        # Once started, execution is independent of base speed. A walk request
        # finishes the checked return path; the existing measured stow gate releases legs.
        self.sample+=self.running.long()
        finished=self.sample>=self.lengths[self.path_id]
        self.sample[finished]=0;self.running[finished]=False
        self.path_id[finished]=torch.multinomial(self.weights,int(finished.sum()),replacement=True) if finished.any() else self.path_id[finished]
        self.clock+=self._env.step_dt
        self.metrics['task_progress'].copy_(self.sample/self.lengths[self.path_id])
        self.metrics['tracking_error'].copy_((self.robot.data.joint_pos[:,self.ids]-self.target).abs().amax(-1))

@configclass
class TaskArmCommandCfg(CommandTermCfg):
    class_type: type=TaskArmCommand
    asset_name: str='robot'
    stage: int=0
    hold_seconds: float=1.
    resampling_time_range: tuple[float,float]=(100000.,100000.)


def task_payload(env,env_ids,standing_mass_max=0.):
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    flat=is_env_assigned_to_terrain(env,'flat')[ids]
    randomize_tcp_payload(env,ids[~flat])
    randomize_tcp_payload(env,ids[flat],mass_range=(0.,standing_mass_max),empty_probability=1. if standing_mass_max==0 else .2)


def task_clearance_violation(env):
    """Conservative actual-state OBB overlap guard, including head clearance.

    Adjacent housing exclusions are explicit in the geometry snapshot. This is
    a sampled control-rate guard, not certification of hardware collision safety.
    """
    from isaaclab.utils.math import matrix_from_quat
    robot=env.scene['robot']
    if not hasattr(env,'_task_boxes'):
        geom=json.loads((DATA/'usd_geometry.json').read_text());names=[];bounds=[];bodyids=[]
        for name,boxes in geom['boxes'].items():
            for index,box in enumerate(boxes):
                names.append((name,index));bounds.append(box)
                bodyids.append(robot.find_bodies('base' if name=='camera' else name,preserve_order=True)[0][0])
        ignores=[set(p) for p in geom['ignore_pairs']];pairs=[]
        for i,(a,ai) in enumerate(names):
            for j,(b,bi) in enumerate(names[:i]):
                if a==b or {a,b} in ignores:continue
                if (a=='arm_link2' and b=='base' and bi==0) or (b=='arm_link2' and a=='base' and ai==0):continue
                pairs.append((i,j))
        bounds=torch.tensor(bounds,device=env.device);pairs=torch.tensor(pairs,device=env.device)
        env._task_box_names=names
        env._task_boxes=(bodyids,(bounds[:,0]+bounds[:,1])/2,(bounds[:,1]-bounds[:,0])/2,pairs)
    ids,local,half,pairs=env._task_boxes
    rot=matrix_from_quat(robot.data.body_quat_w[:,ids]);center=robot.data.body_pos_w[:,ids]+torch.einsum('nbij,bj->nbi',rot,local)
    i,j=pairs.unbind(-1);ra,rb=rot[:,i],rot[:,j];ca,cb=center[:,i],center[:,j]
    a,b=ra.transpose(-1,-2),rb.transpose(-1,-2)
    cross=torch.cross(a[:,:,:,None,:].expand(-1,-1,-1,3,-1),b[:,:,None,:,:].expand(-1,-1,3,-1,-1),dim=-1).flatten(2,3)
    axes=torch.cat([a,b,cross],dim=2);norm=axes.norm(dim=-1,keepdim=True);valid=norm[...,0]>1e-6;axes=axes/norm.clamp_min(1e-6)
    distance=(axes*(cb-ca)[:,:,None,:]).sum(-1).abs()
    radii=(axes@ra).abs()@half[i,:,None]+(axes@rb).abs()@half[j,:,None]
    separated=((distance>radii[...,0]+.003)&valid).any(-1)
    env._task_overlap_pairs=~separated
    return env._task_overlap_pairs.any(-1)


from isaaclab.sim.utils import clone

@clone
def spawn_task_robot(prim_path,cfg,translation=None,orientation=None,**kwargs):
    """Spawn the current USD plus an invisible collision envelope on the base."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from pxr import UsdGeom,UsdPhysics,Gf
    prim=spawn_from_usd(prim_path,cfg,translation,orientation,**kwargs)
    stage=prim.GetStage();parent=stage.GetPrimAtPath(prim_path+'/base')
    if not parent or not parent.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError('Expected floating robot base rigid body')
    box=UsdGeom.Cube.Define(stage,prim_path+'/base/task_camera_keepout')
    box.CreateSizeAttr(1.);box.AddTranslateOp().Set(Gf.Vec3d(.32532,0,.1085))
    box.AddScaleOp().Set(Gf.Vec3f(.05506,.09001,.045))
    box.CreateVisibilityAttr('invisible');UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    # The closed finger CADs overlap (measured ~160 N at rest). Match the
    # existing cuRobo sibling-finger exclusion; retain all other arm collisions.
    finger=stage.GetPrimAtPath(prim_path+'/arm_link7')
    UsdPhysics.FilteredPairsAPI.Apply(finger).CreateFilteredPairsRel().AddTarget(prim_path+'/arm_link8')
    return prim


def standing_roll_centering(env):
    """Keep margin below the 5-degree roll goal while permitting fore/aft pitch."""
    base=env.command_manager.get_term('base_velocity')
    standing=(base.command==0).all(-1)
    return (env.scene['robot'].data.projected_gravity_b[:,1]/0.08715574275).square()*standing
