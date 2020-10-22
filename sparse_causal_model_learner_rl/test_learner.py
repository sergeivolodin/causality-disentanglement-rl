import vectorincrement
import os
import gin
import sparse_causal_model_learner_rl.learner as learner
import sparse_causal_model_learner_rl.config as config


def test_learn_vectorincrement():
    ve_config_path = os.path.join(os.path.dirname(vectorincrement.__file__), 'config', 've5.gin')
    learner_config_path = os.path.join(os.path.dirname(learner.__file__), 'configs', 'test.gin')
    print(ve_config_path, learner_config_path)

    gin.parse_config_file(ve_config_path)
    gin.parse_config_file(learner_config_path)
    l = learner.Learner(config.Config())
    l.train()
    gin.clear_config()