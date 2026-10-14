"""Pure tensor rigid-payload checks; run without Isaac Sim."""
import importlib.util
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[1]/'source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/unified_contract.py'
s=importlib.util.spec_from_file_location('contract',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
mass=torch.tensor([1.,1.]);payload=torch.tensor([0.,1.]);offset=torch.tensor([[.1,0,0],[.1,0,0]])
inertia=torch.eye(3).repeat(2,1,1)
total,result=m.combine_payload_inertia(mass,inertia,payload,offset,radius=0.)
torch.testing.assert_close(total,torch.tensor([1.,2.]))
torch.testing.assert_close(result[0],inertia[0]);torch.testing.assert_close(result[1],torch.diag(torch.tensor([1.,1.005,1.005])))
assert (torch.linalg.eigvalsh(result)>0).all()
print('PASS: zero payload identity, total mass, parallel-axis inertia and positive definiteness')
