#!/bin/bash

set -e

# Disabling GPU
export CUDA_VISIBLE_DEVICES=-1

# launching mongodb
if  [ "$(netstat -tulpn|grep 27017|wc -l)" == "0" ]
then
	echo "Starting mongo"
	sudo service mongodb start
fi

# script directory
# https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "Running tests..."
pytest causal_util sparse_causal_model_learner_rl vectorincrement keychest gin_tune

rm -rf $DIR/results/test_ve5

echo "Running training with gin/sacred/tune"
python -m sparse_causal_model_learner_rl.learner --config $DIR/sparse_causal_model_learner_rl/configs/test.gin --config $DIR/vectorincrement/config/ve5.gin

rm -rf $DIR/results/test_ve5

# final message
echo "All tests PASSED!"
