"""Check initialization is per-environment and leaves original tasks unchanged."""
import importlib.util,sys,types
from pathlib import Path
import gymnasium as gym
import torch
# This wrapper's camera helper is not involved in the test.
for name in ['isaaclab','isaaclab.utils','isaaclab.utils.math']:sys.modules[name]=types.ModuleType(name)
s=importlib.util.spec_from_file_location('rl_utils',Path(__file__).resolve().parents[1]/'scripts/reinforcement_learning/rl_utils.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Env(gym.Env):
    num_envs=2;step_dt=.02
    def __init__(self,hold):self.cfg=types.SimpleNamespace(startup_hold_seconds=hold);self.tick=0
    def reset(self,**kwargs):self.tick=0;return {},{}
    def step(self,a):
        self.last=a.clone();self.tick+=1
        return {},torch.zeros(2),torch.tensor([self.tick==5,False]),torch.zeros(2,dtype=torch.bool),{}
for hold in [0.,.06]:
    e=Env(hold);w=m.ActionDelayWrapper(e,delay_steps_range=(1,1));w.reset();seen=[]
    for i in range(10):w.step(torch.ones(2,12));seen.append(e.last[:,0].tolist())
    if hold:
        assert all(row==[0.,0.] for row in seen[:3]);assert seen[3]==[1.,1.]
        assert all(row==[0.,1.] for row in seen[5:8]);assert seen[8]==[1.,1.]
    else:assert seen[0]==[0.,0.] and seen[1]==[1.,1.]
print('PASS: per-environment startup hold, partial reset, original task behavior')
