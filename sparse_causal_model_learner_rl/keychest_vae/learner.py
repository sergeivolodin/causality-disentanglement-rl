import gin
import scipy.misc
from sparse_causal_model_learner_rl.learners.abstract_learner import AbstractLearner
from sparse_causal_model_learner_rl.keychest_vae.data import learner, h, w, c, engine, plot_data, get_dataloader
from sparse_causal_model_learner_rl.keychest_vae.vae import ObsNet, ObsModel
from causal_util.helpers import postprocess_info
import gin
import numpy as np
from matplotlib import pyplot as plt
import torchvision
from torch.autograd import Variable
import os
from PIL import Image

def write_np_img(data, fn):
    rescaled = (255.0 * data).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save(fn)

@gin.register
class VAEKeyChestLearner(AbstractLearner):
    """Learn autoencoder on data from an environment."""
    def __init__(self, config, callback=None):
        super().__init__(config, callback)
        self.train_loader = get_dataloader(self.config.get('train_samples'))
        self.eval_loader = get_dataloader(self.config.get('eval_samples'))
        self.m = ObsModel(self.train_loader, self.eval_loader)

    def __getstate__(self):
        return {}

    def __setstate__(self, z):
        pass

    def _epoch(self):
        epoch_info = self.m.train()

        if self.epochs % self.config.get('eval_every', 20) == 0:
            epoch_info.update(self.m.eval())

        # process epoch information
        epoch_info = postprocess_info(epoch_info)
        epoch_info['epochs'] = self.epochs

        # send information downstream
        if self.callback:
            self.callback(self, epoch_info)

        # update config
        self.config.update(epoch_info=epoch_info)
        self.epochs += 1

        self.epoch_info = epoch_info

        return epoch_info

    def maybe_write_artifacts(self, path_epoch, add_artifact_local):

        if self.epochs % self.config.get('image_every', 5) == 0:

            def get_images(m, loader):
                self.m.model.eval()
                _, (states, actions, next_states) = next(enumerate(loader))
                states = Variable(states).cuda()
                actions = Variable(actions).cuda()
                next_states = Variable(next_states).cuda()

                # mnist
                # states = next(enumerate(m.train_loader))[1][0].cuda()
                # actions = torch.zeros((states.shape[0], 4)).cuda()
                # next_states = states
            
                out, _, _ = self.m.model(states, actions)
                out = np.rollaxis(out.detach().cpu().numpy(), 1, 4)
                return out


            train_images = get_images(self.m, self.train_loader)
            eval_images = get_images(self.m, self.eval_loader)
            demo_images = np.rollaxis(self.train_loader.dataset.tensors[0].numpy(), 1, 4)

            train_image_idx = np.random.choice(range(len(train_images)))
            eval_image_idx = np.random.choice(range(len(eval_images)))
            demo_image_idx = np.random.choice(range(len(demo_images)))


            train_image = train_images[train_image_idx]
            eval_image = eval_images[eval_image_idx]
            demo_image = demo_images[demo_image_idx]

            os.makedirs(path_epoch, exist_ok=True)
            with path_epoch:
                write_np_img(train_image, 'train_image.png')
                write_np_img(eval_image, 'eval_image.png')
                write_np_img(demo_image, 'demo_image.png')

                add_artifact_local(path_epoch / "train_image.png")
                add_artifact_local(path_epoch / "eval_image.png")
                add_artifact_local(path_epoch / "demo_image.png")
