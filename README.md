## Learning Interpretable Abstract Representations in Reinforcement Learning via Model Sparsity

[![Build Status](https://travis-ci.com/sergeivolodin/causality-disentanglement-rl.svg?branch=master)](https://travis-ci.com/sergeivolodin/causality-disentanglement-rl)
![Python application](https://github.com/sergeivolodin/causality-disentanglement-rl/workflows/Python%20application/badge.svg)

The problem of learning good abstractions is arguably one of the cornerstones of Artificial Intelligence. One of the theoretical or philosophical approaches to learn abstract representations is the Consciousness Prior proposed by Yoshua Bengio. One of the key components in that project is the sparsity of the transition model, which supposedly leads to good learned abstractions. In this project, we design a simple environment where abstractions can be learned. We propose a practical framework for learning abstractions via sparsity of the transition model. The results show that we are able to recover the correct representation. We provide theoretical formulation of the problem and the explanation of the results. We provide exciting future research directions and concrete questions in the domain of learning good abstractions.

Talk for <a href="http://mlss.tuebingen.mpg.de/2020/">MLSS 2020</a>:<br />
[![Learning Interpretable Abstract Representations in Reinforcement Learning via Model Sparsity
](https://img.youtube.com/vi/4cO4XU15isQ/0.jpg)](https://www.youtube.com/watch?v=4cO4XU15isQ "Learning Interpretable Abstract Representations in Reinforcement Learning via Model Sparsity")

Overall idea:<br />
<img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/CausalityConcepts.pdf.png" />

Proposed architecture:<br />
<img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/CausalLearner.pdf.png" height="300" />

Original causal graph (left) and optimized causal graph (right):<br />
<img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/graph_orig.pdf.png" height="300" />
<img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/graph_sparse.pdf.png" height="300" />


Done as a semester project at Laboratory of Computational Neuroscience at the Swiss Federal Institute of Technology in Lausanne (EPFL)
<a href="https://www.overleaf.com/read/nqgjrjbcybrp">See full project report</a>

<a href="https://www.overleaf.com/read/mmmbrrvnkffq">Master thesis report draft</a>

We use [pytorch](https://pytorch.org/) to learn the sparse model and [stable baselines](https://stable-baselines.readthedocs.io/en/master/) for RL.

### Installation
1. You will need `conda` and `pip`
2. Install requirements: `pip install -r requirements.txt`
3. Install gin_tune: `pip install -e gin_tune`
3. Set up a MongoDB database `test` on port `27017` on local machine
4. Having ray installed, run `python ray/python/ray/setup-dev.py` to patch your ray installation
4. `pip install -e .`

### Performance of envs
1. `python -m causal_util.env_performance --env KeyChest-v0 --config keychest/config/5x5.gin` 
2. `python -m causal_util.env_performance --env CartPole-v0`
3. `python -m causal_util.env_performance --env VectorIncrement-v0 --config vectorincrement/config/ve5.gin`

### Learner
`python -m sparse_causal_model_learner_rl.learner --config $(pwd)/sparse_causal_model_learner_rl/configs/test_tune.gin --config $(pwd)/vectorincrement/config/ve5.gin`

### More images

Non-convex losses during training:<br />
<img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/descent_pca.pdf.png" height="200" />
<img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/plot_nodata_synthetic.png" height="200" />

Training feedback:<br />
<img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/sml.pdf.png" height="300" />

Tensorboard integration:<br />
<img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/vectorincrement_tb.png" height="300" />

KeyChest environment:<br />
<img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/keychest.gif" />

VectorIncrement environment:<br />
<img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/env_ve5_noperm.gif" height="200" /> <img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/env_ve5_perm.gif" height="200" />


## VectorIncrement with 5 components
<img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/ve5_gifs/CausalModel.png_all.gif-opt.gif" height="200" /> <img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/ve5_gifs/model.png_all.gif-opt.gif" height="200" />

<img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/ve5_gifs/metricsnnz.png_all.gif-opt.gif" height="100" /> <img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/ve5_gifs/lossesopt1sparsityvalue.png_all.gif-opt.gif" height="100" /> <img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/ve5_gifs/metricslastmultsparsitygap.png_all.gif-opt.gif" height="100" /> <img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/ve5_gifs/metricsfitlossfcons.png_all.gif-opt.gif" height="100" /> <img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/ve5_gifs/metricsfitrecfitaccloss01agg.png_all.gif-opt.gif" height="100" /> <img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/ve5_gifs/lossesopt1fitvalue.png_all.gif-opt.gif" height="100" />

### Sanity checks


Gumbel model on ve5 (features raw):
`python -m sparse_causal_model_learner_rl.learner --config vectorincrement/config/ve5_nonlinear.gin --config sparse_causal_model_learner_rl/configs/rec_nonlin_gnn_gumbel
 --nofail --n_gpus 0 --n_cpus 1`
 
#### New modular config with ve2
1. Only features on vectorincrement (new config)
    `python -m sparse_causal_model_learner_rl.learner --config sparse_causal_model_learner_rl/configs/rl_const_sparsity_obs_space.gin --config sparse_causal_model_learner_rl/configs/env_ve2.gin --nofail --n_gpus 1`
   After ~25 minutes (~9000 steps) gives the following graph:
   
   ![](images/ve2_onlyf.png)
2. Features + reward + done
    `python -m sparse_causal_model_learner_rl.learner --nofail --n_gpus 1 --config sparse_causal_model_learner_rl/configs/rl_const_sparsity_obs_space.gin --config sparse_causal_model_learner_rl/configs/env_ve2_with_rew_done.gin`
   After ~2 hours (45k steps) gives the following graph:
   
   ![](images/ve2.png)
3. No decoder/encoder:
    `python -m sparse_causal_model_learner_rl.learner --config sparse_causal_model_learner_rl/configs/rl_const_sparsity_obs_space.gin --config sparse_causal_model_learner_rl/configs/env_ve2_raw_with_rew_done.gin --nofail --n_gpus 1`
   After ~45 minutes gives:
   
   ![](images/ve2_nodec.png)
4. ve5 with no encoder:
   `python -m sparse_causal_model_learner_rl.learner --nofail --n_gpus 1 --config sparse_causal_model_learner_rl/configs/rl_const_sparsity_obs_space.gin --config sparse_causal_model_learner_rl/configs/env_ve5_raw_with_rew_done.gin`
   15 minutes (~8k steps)
   ![](images/ve5_raw.png)
5. ve5 with decoder/encoder
   `python -m sparse_causal_model_learner_rl.learner --nofail --n_gpus 1 --config sparse_causal_model_learner_rl/configs/rl_const_sparsity_obs_space.gin --config sparse_causal_model_learner_rl/configs/env_ve5_with_rew_done.gin`
   4 hours (25000 epochs):
   
   ![](images/ve5.png)
5. ve5 without encoder, adaptive Lagrange sparsity:
   `python -m sparse_causal_model_learner_rl.learner --nofail --n_gpus 1 --config sparse_causal_model_learner_rl/configs/rl_const_sparsity_obs_space.gin --config sparse_causal_model_learner_rl/configs/env_ve5_raw_with_rew_done.gin --config sparse_causal_model_learner_rl/configs/with_lagrange_dual_sparsity.gin`
   2 hours (50k steps)
   ![](images/ve5_raw_lagrange.png) ![](images/ve5_raw_lagrange_graph.png)
5. SparseMatrix (size 5x5, 7 elements). 3mins or ~3000 epochs, Lagrange

   `python -m sparse_causal_model_learner_rl.learner --nofail --n_gpus 1 --config sparse_causal_model_learner_rl/configs/rl_const_sparsity_obs_space.gin --config sparse_causal_model_learner_rl/configs/env_sm5_noact.gin --config sparse_causal_model_learner_rl/configs/with_lagrange_dual_sparsity.gin`
6. SparseMatrix (size 5x5, 7 elements + actions). 10mins or ~400 epochs (a single epoch contains more interations), Lagrange:
   `python sparse_causal_model_learner_rl/learner.py --nofail --n_gpus 1 --config sparse_causal_model_learner_rl/configs/rl_const_sparsity_obs_space.gin --config sparse_causal_model_learner_rl/configs/env_sm5.gin --config sparse_causal_model_learner_rl/configs/with_lagrange_dual_sparsity.gin` (replace last config with `sparse_causal_model_learner_rl/configs/with_lagrange_dual_sparsity_per_component.gin` for per-component sparsity constraints)
7. ve5 with a decoder and adaptive sparsity (45000 epochs, ~4-5 hours):
   `python sparse_causal_model_learner_rl/learner.py --n_gpus 1 --nofail --config sparse_causal_model_learner_rl/configs/rl_const_sparsity_obs_space.gin --config sparse_causal_model_learner_rl/configs/env_ve5_with_rew_done.gin --config sparse_causal_model_learner_rl/configs/with_lagrange_dual_sparsity.gin`
8. sm5_linear:
   `python -m sparse_causal_model_learner_rl.learner --nofail --n_gpus 1 --config sparse_causal_model_learner_rl/configs/rl_const_sparsity_obs_space.gin --config sparse_causal_model_learner_rl/configs/env_sm5_linear.gin --config sparse_causal_model_learner_rl/configs/with_lagrange_dual_sparsity_per_component.gin`

#### KeyChest
1. PPO on KeyChest: `python sb.py --evaluate --train_steps 5000000 --config ../keychest/config/5x5.gin --trainer DQN`
2. DQN on KeyChest: `python sb.py --config ../keychest/config/5x5.gin --evaluate --train_steps 5000000 --train`

Success of DQN shows that the environment is Markov:

<img src="https://github.com/sergeivolodin/causality-disentanglement-rl/blob/master/images/dqn_ppo_keychest.png" width="300" />
