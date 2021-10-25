import gin
import numpy as np


@gin.configurable
def save_data(obs, now_epoch_info, **kwargs):
    epochs = now_epoch_info['epochs']
    obs = obs.detach().cpu().numpy()
    print(epochs, obs.shape, obs)
    with open("/media/output/data-pickle/epoch%010d.pkl" % epochs, "wb") as f:
        np.save(f, obs)
