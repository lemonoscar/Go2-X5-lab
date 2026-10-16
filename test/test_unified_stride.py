"""Contact-event regression without importing the Isaac application."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import torch

class ManagerTermBase:
    def __init__(self, cfg, env):
        pass

source = Path(__file__).resolve().parents[1] / 'source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/unified_stride.py'
module = ast.parse(source.read_text())
module.body = [node for node in module.body if isinstance(node, ast.ClassDef)]
namespace = dict(torch=torch, ManagerTermBase=ManagerTermBase, quat_apply_inverse=lambda q, v: v)
exec(compile(module, str(source), 'exec'), namespace)
robot = NS(find_bodies=lambda *a, **k: ([0,1,2,3], None), data=NS(
    root_pos_w=torch.zeros(1,3), root_quat_w=torch.tensor([[1.,0,0,0]]),
    body_pos_w=torch.zeros(1,4,3), projected_gravity_b=torch.tensor([[0.,0.,-1.]])))
sensor = NS(find_bodies=robot.find_bodies, data=NS(net_forces_w=torch.zeros(1,4,3)))
term = NS(command=torch.tensor([[.5,0,0]]))
env = NS(scene={'robot':robot,'contact_forces':sensor}, num_envs=1, device='cpu', step_dt=.02,
         command_manager=NS(get_term=lambda name: term))
cost = namespace['ShortStepCost'](None, env)
def step(contact, x=0.):
    sensor.data.net_forces_w[:] = 0
    sensor.data.net_forces_w[:,:,2] = 20. if contact else 0.
    robot.data.body_pos_w[:,:,0] = x
    return float(cost(env))
def cycle(air_steps, excursion):
    cost.reset()
    for _ in range(20): step(True)
    for _ in range(air_steps): step(False)
    return step(True, excursion)
assert cycle(3,.03) > cycle(10,.18)
assert cycle(10,.18) == 0
assert cycle(10,.03) > 0
term.command.zero_()
assert cycle(2,.01) == 0
term.command[0,2] = .2
assert cycle(10,0.) == 0  # Pure yaw has no translational stride requirement.
cost.reset(); assert step(True) == 0  # No fabricated startup touchdown.
# Schmitt trigger: intermediate forces retain the preceding contact state.
sensor.data.net_forces_w[:,:,2] = 5.; cost(env); assert cost.contact.all()
sensor.data.net_forces_w[:,:,2] = 2.; cost(env); assert not cost.contact.any()
sensor.data.net_forces_w[:,:,2] = 5.; cost(env); assert not cost.contact.any()
print('PASS: short/long swings, excursion, standing/yaw gates, reset and contact hysteresis')
