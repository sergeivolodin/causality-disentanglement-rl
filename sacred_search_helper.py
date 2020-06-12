import numpy as np
import itertools
import os

sweep_02 = {
    #s 2 3 5 10
    'p_ord': np.linspace(0.1, 1, 2),
    'eps_dinv': np.logspace(-2, 2, 2),
}

sweep = sweep_02

n_cpus = 2

# all parameters as a list
all_params = list(itertools.product(*[[(x, z) for z in y] for x, y in sweep.items()]))
all_params = [dict(t) for t in all_params]
print("Total runs: %d on %d CPUs" % (len(all_params), n_cpus))

def process_i(i):
    """Process i'th parameter."""
    param = all_params[i]
    print(i, param)
    
    config_str = ' '.join(["'%s=%s'" % (str(k), str(v)) for k, v in param.items()])
    
    cmd = "python sacred_sparse_reinforce.py --force with %s" % config_str
    print(cmd)
    os.system(cmd)