import vectorincrement
import os
import gin
import sparse_causal_model_learner_rl.learners.rl_learner as learner
import sparse_causal_model_learner_rl.learners.abstract_learner as abstract_learner
import sparse_causal_model_learner_rl.config as config
import pytest
from sparse_causal_model_learner_rl.sacred_gin_tune.sacred_wrapper import load_config_files


def test_learn_vectorincrement():
    ve_config_path = os.path.join(os.path.dirname(vectorincrement.__file__), 'config', 've5.gin')
    learner_config_path = os.path.join(os.path.dirname(learner.__file__), '..', 'configs', 'test.gin')
    print(ve_config_path, learner_config_path)

    load_config_files([ve_config_path, learner_config_path])

    l = learner.CausalModelLearnerRL(config.Config())
    l.train()
    gin.clear_config()

def test_abstract_learner_create():
    f = os.path.join(os.path.dirname(abstract_learner.__file__), '..', 'configs', 'base_learner.gin')
    load_config_files([f])
    l = abstract_learner.AbstractLearner(config.Config())
    l.train()
    gin.clear_config()

@pytest.fixture(autouse=True)
def clean_gin():
    gin.clear_config()
    yield
    gin.clear_config()