## Learning Interpretable Abstract Representations in Reinforcement Learning via Model Sparsity

<img src="https://api.travis-ci.com/sergeivolodin/causality-disentanglement-rl.svg?branch=master&status=started" />

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

We use torch to learn the sparse model and stable baselines for RL.

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
