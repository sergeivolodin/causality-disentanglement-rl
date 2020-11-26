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
MODULE_PATH=$(python3 -c "import sparse_causal_model_learner_rl; print(sparse_causal_model_learner_rl.__path__[0])")

echo "Running tests..."
python3 -m pytest causal_util sparse_causal_model_learner_rl vectorincrement keychest gin_tune

rm -rf $DIR/results/test_ve5

echo "Running training with gin/sacred/tune"
python3 -m sparse_causal_model_learner_rl.learner --config $MODULE_PATH/configs/test.gin --config $MODULE_PATH/../vectorincrement/config/ve5.gin --nofail

rm -rf $DIR/results/test_ve5

# final message
echo "All tests PASSED!"
