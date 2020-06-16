import numpy as np
import itertools
import os

sweep_02 = {
    'v_n': [2, 3, 5, 10],
    #'p_ord': np.linspace(0.1, 1, 10),
    #'eps_dinv': np.logspace(-2, 2, 10),
    'v_seed': [43, 42, 1, 5],
    #'d_init_randomness': np.linspace(0.5, 10, 5),
    'repetitions': range(10)
}

sweep = sweep_02

n_cpus = 24

# all parameters as a list
all_params = list(itertools.product(*[[(x, z) for z in y] for x, y in sweep.items()]))
all_params = [dict(t) for t in all_params]
print("Total runs: %d on %d CPUs" % (len(all_params), n_cpus))

def process_i(i):
    """Process i'th parameter."""
    param = all_params[i]
    if 'v_n' in param:
        param['v_k'] = param['v_n']
    print(i, param)
    
    config_str = ' '.join(["'%s=%s'" % (str(k), str(v)) for k, v in param.items()])
    
    cmd = "python sacred_sparse_reinforce.py --force with %s" % config_str
    print(cmd)
    os.system(cmd)