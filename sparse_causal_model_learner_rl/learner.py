import argparse

import matplotlib as mpl

mpl.use('Agg')

import logging
import ray
import os
import gin
import sys
from sparse_causal_model_learner_rl.config import Config
from sparse_causal_model_learner_rl.sacred_gin_tune.sacred_wrapper import load_config_files, main_fcn, learner_gin_sacred
from causal_util.helpers import set_default_logger_level

parser = argparse.ArgumentParser(description="Causal learning experiment")
parser.add_argument('--config', type=str, required=True, action='append')
parser.add_argument('--n_cpus', type=int, required=False, default=None)
parser.add_argument('--print_config', action='store_true')
parser.add_argument('--n_gpus', type=int, required=False, default=None)
parser.add_argument('--nowrap', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--ray_debug_timeline', type=str, required=False, default=None)
parser.add_argument('--nofail', help="Disable killing ray actors at the end of the trial", action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    cwd = os.getcwd()
    config = args.config
    config = [c if os.path.isabs(c) else os.path.join(cwd, c) for c in config]
    logging.info(f"Absolute config paths: {config}")

    if args.print_config:
        load_config_files(config)
        gin.finalize()
        print(gin.config_str())
        sys.exit(0)

    if args.nowrap:
        # useful for debugging/testing
        set_default_logger_level(logging.DEBUG)
        load_config_files(config)
        gin.finalize()
        config = Config()

        main_fcn(config=config, ex=None, checkpoint_dir=None, do_tune=False, do_sacred=False,
                 do_tqdm=True, do_exit=False)
    else:
        kwargs = {'num_cpus': args.n_cpus}
        if args.n_cpus == 0:
            kwargs = {'num_cpus': 1, 'local_mode': True}
        ray.init(**kwargs, num_gpus=args.n_gpus, include_dashboard=True)

        if args.resume:
            gin.bind_parameter('tune_run.resume', True)

        learner_gin_sacred(config, nofail=args.nofail)

        if args.ray_debug_timeline:
            ray.timeline(filename=args.ray_debug_timeline)