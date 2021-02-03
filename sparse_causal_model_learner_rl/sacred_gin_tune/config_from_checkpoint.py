import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
from causal_util.helpers import CPU_Unpickler
import argparse
import gin


parser = argparse.ArgumentParser(description="Print gin configuration given a checkpoint")
parser.add_argument('--checkpoint', type=str, required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.checkpoint, 'rb') as f:
        learner = CPU_Unpickler(f).load()

    output_file = f"{args.checkpoint}.gin"

    with open(output_file, 'w') as f:
        f.write(gin.config_str())

    print("Configuration written to", output_file)