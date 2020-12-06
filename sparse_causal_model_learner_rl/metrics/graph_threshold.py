from sparse_causal_model_learner_rl.visual.learner_visual import select_threshold
import gin

@gin.configurable
def threshold_action(model, **kwargs):
    return select_threshold(model.Ma, do_plot=False)

@gin.configurable
def threshold_features(model, **kwargs):
    return select_threshold(model.Mf, do_plot=False)
