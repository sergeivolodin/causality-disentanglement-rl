import argparse
import os
import uuid

import gin
from observation_encoder import KerasEncoderWrapper

from causal_util import load_env

parser = argparse.ArgumentParser(description="Create a KerasEncoder, save weights")
parser.add_argument('--config', type=str, required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    config_basename = "none"
    if args.config:
        gin.parse_config_file(args.config)
        gin.bind_parameter("observation_encoder.KerasEncoder.model_filename", None)
        config_basename = os.path.basename(args.config)[:-4]
    os.makedirs('encoders', exist_ok=True)
    out_fn = f"encoders/encoder-config-{config_basename}-{str(uuid.uuid1())}.pb"

    env = load_env()
    wrapped = KerasEncoderWrapper(env)
    wrapped.f.save(out_fn)
    print(f"Saved to {out_fn}")
