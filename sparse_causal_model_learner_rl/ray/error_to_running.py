import argparse
import os
import json


parser = argparse.ArgumentParser(description="Replace status of errored trials to 'running'")
parser.add_argument('--experiment_dir', help="Directory with experiments", type=str, required=True)


if __name__ == '__main__':
    args = parser.parse_args()
    exp_dir = args.experiment_dir
    print(f"Running in {exp_dir}")
    exp_states = sorted([os.path.join(exp_dir, x) for x in os.listdir(exp_dir) if x.startswith("experiment_state")])
    for exp_state in exp_states:
        with open(exp_state, 'r') as f:
            state = json.load(f)
        os.rename(exp_state, f"{os.path.dirname(exp_state)}/._orig_{os.path.basename(exp_state)}.orig")
        for ckpt in state['checkpoints']:
            if ckpt['status'] == 'ERROR':
                print("Overwriting ERROR to RUNNING")
                ckpt['status'] = 'RUNNING'
        with open(exp_state, 'w') as f:
            json.dump(state, f)
