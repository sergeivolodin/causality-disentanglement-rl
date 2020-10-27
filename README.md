## Learning Interpretable Abstract Representations in Reinforcement Learning via Model Sparsity

The problem of learning good abstractions is arguably one of the cornerstones of Artificial Intelligence. One of the theoretical or philosophical approaches to learn abstract representations is the Consciousness Prior proposed by Yoshua Bengio. One of the key components in that project is the sparsity of the transition model, which supposedly leads to good learned abstractions. In this project, we design a simple environment where abstractions can be learned. We propose a practical framework for learning abstractions via sparsity of the transition model. The results show that we are able to recover the correct representation. We provide theoretical formulation of the problem and the explanation of the results. We provide exciting future research directions and concrete questions in the domain of learning good abstractions.

Done as a semester project at Laboratory of Computational Neuroscience at the Swiss Federal Institute of Technology in Lausanne (EPFL)
<a href="https://www.overleaf.com/read/nqgjrjbcybrp">See full project report</a>

We use torch to learn the sparse model and stable baselines for RL.

### Installation
1. You will need `conda` and `pip`
2. Install requirements: `pip install -r requirements.txt`
3. Install gin_tune: `pip install -e gin_tune`
3. Set up a MongoDB database `test` on port `27017` on local machine
4. `pip install -e .`

### Performance of envs
1. `python -m causal_util.env_performance --env KeyChest-v0 --config keychest/config/5x5.gin` 
2. `python -m causal_util.env_performance --env CartPole-v0`
3. `python -m causal_util.env_performance --env VectorIncrement-v0 --config vectorincrement/config/ve5.gin`
