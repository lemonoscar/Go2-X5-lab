"""Run directly with a Torch-enabled interpreter; no simulator required."""
import importlib.util
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('contract', ROOT / 'source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/unified_contract.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def main():
    x = torch.tensor([[.1, -.05, .05], [-.1, .05, -.05], [.75, 0, 0], [-.5, 0, 0],
                      [0, .2, 0], [0, 0, -.3], [.08, .15, -.03], [.75, .2, .3]])
    y = c.filter_command(x)
    torch.testing.assert_close(y[:2], torch.zeros(2, 3))
    torch.testing.assert_close(y[2:6], x[2:6])
    torch.testing.assert_close(y[6], torch.tensor([0, .15, 0]))
    torch.testing.assert_close(y[-1], x[-1] / 3**.5)
    random = c.filter_command(torch.randn(10000, 3))
    limits = torch.tensor([.75, .2, .3]).expand_as(random).clone()
    limits[:, 0] = torch.where(random[:, 0] < 0, .5, .75)
    assert ((random/limits).square().sum(-1) <= 1.000001).all()
    assert ((random == 0) | (random.abs() > torch.tensor([.1,.05,.05]))).all()
    torch.testing.assert_close(c.filter_command(random), random)
    try:
        c.filter_command(torch.tensor([[float('nan'), 0, 0]]))
    except ValueError:
        pass
    else:
        raise AssertionError('NaN was accepted')
    h = c.ShortHistory(2, 'cpu')
    f = torch.ones(2,54)
    assert (h.update(f, 0) == 1).all()
    assert (h.update(f*2, 1) == 1).all()
    out = h.update(f*3, 2).clone()
    assert (out.reshape(2,4,54)[:,-1] == 2).all()
    torch.testing.assert_close(h.update(f*999, 2), out)
    h.reset(torch.tensor([0]))
    out = h.update(f*4, 3).reshape(2,4,54)
    assert (out[0] == 4).all() and (out[1,-1] == 3).all()
    h.reset(torch.tensor([0]))
    assert (h.update(f*7, 3).reshape(2,4,54)[0] == 7).all()
    print('PASS: dead zones, signed limits, budget, idempotence, nonfinite rejection, history ordering and partial reset')


if __name__ == '__main__':
    main()
