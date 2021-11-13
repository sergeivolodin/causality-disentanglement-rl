import gin
import torch


@gin.configurable
def contrastive_loss_full(obs, decoder, epoch_profiler, margin_val=0.1, **kwargs):
  """Compute the margin loss excluding elements where it is exactly 0.

     Args:
       features: input tensor (batch, features)
       margin_val: the value of the margin
     Returns:
       sum of max(0, margin - pairwise_distances) where distances > 0
         divided by |batch|
  """
  features = decoder(obs)

  epoch_profiler.start('contrastive_full')

  epoch_profiler.start('cdist')
  # approximate pairwise distances with matrix multiplication
  features_pairwise_approx = torch.cdist(features, features, p=2)
  epoch_profiler.end('cdist')

  epoch_profiler.start('eye')
  # identity matrix
  eye = torch.eye(features_pairwise_approx.shape[0],
                  dtype=torch.uint8,
                  device=features_pairwise_approx.device)
  epoch_profiler.end('eye')

  epoch_profiler.start('where')
  # off-diagonal margin where distance > 0
  # the values are inexact because of matrix multiplication
  deltas = torch.where((features_pairwise_approx > 0) & (eye < 0.5),
                        torch.nn.ReLU()(margin_val - features_pairwise_approx),
                        torch.zeros((), device=features_pairwise_approx.device))
  epoch_profiler.end('where')

  epoch_profiler.start('nonzero')
  # indexes of non-zero elements
  nonzero_x, nonzero_y = torch.nonzero(deltas, as_tuple=True)
  epoch_profiler.end('nonzero')

  # computing exact value in case if it's feasible (guaranteed linear time)
  if len(nonzero_x) <= len(features):
    epoch_profiler.start('exact')
    # pairwise distances for selected items
    distances = torch.norm(features[nonzero_x] - features[nonzero_y],
                           p='fro', dim=1)

    # only selecting non-0 distances
    distances = distances[distances > 0]

    # computing the exact margin
    loss = torch.nn.ReLU()(margin_val - distances).sum()

    # exact number of non-zero elements
    n_nonzero_exact = len(distances)
    epoch_profiler.end('exact')
  else:
    # inexact loss
    loss = deltas.sum()

    # no exact number
    n_nonzero_exact = -1

  epoch_profiler.end('contrastive_full')

  return {'loss': loss / features.shape[0],
          'metrics': {
              'n_nonzero_approx': len(nonzero_x),
              'n_nonzero_exact': n_nonzero_exact
          }}
