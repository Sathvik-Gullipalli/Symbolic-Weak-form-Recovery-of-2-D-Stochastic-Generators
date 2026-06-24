import numpy as np
import sys
sys.path.insert(0, '/Users/admin/2d_sde/src')
from sde2d.generator import fit_generator_2d
from sde2d.systems import REGISTRY

def test_multi_lag():
    sys_obj = REGISTRY["indep_ou"].cls()
    x = sys_obj.simulate(dt=0.01, M=1000, seed=42)
    cur = x[:-2]
    inc_stack = np.zeros((len(cur), 2, 2))
    inc_stack[:, 0, :] = x[1:-1] - cur
    inc_stack[:, 1, :] = x[2:] - cur
    
    fit2 = fit_generator_2d(cur, inc_stack, dt=0.01, drift_lags=(1, 2), lag_bias_correct=False)
    print("Multi-lag fit drift:", fit2.drift)

if __name__ == "__main__":
    test_multi_lag()
