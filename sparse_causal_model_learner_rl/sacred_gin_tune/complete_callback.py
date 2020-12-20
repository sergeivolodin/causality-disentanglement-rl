from ray.tune.trial_runner import Callback
from ray.tune import Stopper
import gin

@gin.configurable
def get_singleton(cls, name):
    """Return same object for a fixed name."""
    if not hasattr(get_singleton, '_cache'):
        get_singleton._cache = {}
    if name not in get_singleton._cache:
        get_singleton._cache[name] = cls()
    return get_singleton._cache[name]

@gin.configurable
class CompletesCallback(Callback):
    """Count completed trials."""

    def __init__(self, *args, **kwargs):
        self.completed_trials = 0
        super(CompletesCallback, self).__init__(*args, **kwargs)

    def on_trial_complete(self, iteration, trials, trial, **kwargs):
        #print("Completed trial", trial)
        self.completed_trials += 1


@gin.configurable
class StopOnCompletes(Stopper):
    """Stop on a given number of completed trials."""
    def __init__(self, completes_callback, target_trials=10, *args, **kwargs):
        super(StopOnCompletes, self).__init__(*args, **kwargs)
        self.target_trials = target_trials
        self.completes_callback = completes_callback

    def __call__(self, *args, **kwargs):
        return False

    def stop_all(self):
        #print("stop_all call", self.completes_callback.completed_trials, self.target_trials)
        return self.completes_callback.completed_trials >= self.target_trials
