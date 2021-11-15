from sparse_causal_model_learner_rl.trainable.decoder import SingleObjectPerChannelDecoder
import torch
from torch.testing import assert_close
import gin


def test_single_object():
    gin.bind_parameter('SingleObjectPerChannelDecoder.input_shape', [4, 3, 2])
    gin.bind_parameter('SingleObjectPerChannelDecoder.output_shape', (6,))

    dec = SingleObjectPerChannelDecoder()
    obs = torch.zeros((1, 4, 3, 2), dtype=torch.float32)
    obs[0, 1, 2, 0] = 1.0
    obs[0, 3, 1, 1] = 1.0

    out = dec(obs).detach().cpu().numpy()
    assert out.shape == (1, 6)
    # ones f1 f2 x f1 f2 y f1 f2
    #      0  1    2  3    4  5
    out = out[0]
    assert_close(out[0], 1.0)
    assert_close(out[1], 1.0)
    assert_close(out[2], 2.)
    assert_close(out[3], 4.)
    assert_close(out[4], 3.)
    assert_close(out[5], 2.)
