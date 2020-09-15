## Learning Interpretable Abstract Representations in Reinforcement Learning via Model Sparsity

The problem of learning good abstractions is arguably one of the cornerstones of Artificial Intelligence. One of the theoretical or philosophical approaches to learn abstract representations is the Consciousness Prior proposed by Yoshua Bengio. One of the key components in that project is the sparsity of the transition model, which supposedly leads to good learned abstractions. In this project, we design a simple environment where abstractions can be learned. We propose a practical framework for learning abstractions via sparsity of the transition model. The results show that we are able to recover the correct representation. We provide theoretical formulation of the problem and the explanation of the results. We provide exciting future research directions and concrete questions in the domain of learning good abstractions.

Done as a semester project at Laboratory of Computational Neuroscience at the Swiss Federal Institute of Technology in Lausanne (EPFL)
<a href="https://www.overleaf.com/read/nqgjrjbcybrp">See full project report</a>

We use TensorFlow 2.0 with TF.Agents.

### Installation
1. You will need `conda` and `pip`
2. Install requirements: `pip install -r requirements.txt`
3. Set up a MongoDB database `test` on port `27017` on local machine

### Code/notebook files
There are two approaches outlined in the report. We either learn the sparse model directly from data ("X"), or we first learn a model for observations, and then sparsify it ("Y").
Since only the latter option worked in experiments, only this option is implemented for Task A (see below).
The code learning directly from data is not working in task A (see `exp_poc.py` in Other Files).

The code is roughly separated into three tasks.

#### Task A: running RL on VectorIncrement environment with curiosity [only support sparsifying observation model]

1. `sacred_sparse_reinforce.py` -- running model sparsification on RL data, using Sacred
1. `sacred-analysis.ipynb` -- Analysis of Sacred experiments
1. `sacred_search.ipynb` -- Sacred Grid Search starter
1. `environment_vec_pg-linearnet-curiosity-decoder-prior.ipynb` -- policy gradients + sacred

#### Task B: learning abstractions from synthetic data

1. `synthetic_data.py` -- defines the synthetic dataset matching data from VectorIncrement
1. `synthetic_experiment.py` -- run experiment on synthetic data
1. `create_analyze_runs_hyperparam_narrow_search_around_wide_55-sparsity.ipynb` -- grid search for Task B
1. `create_analyze_runs_hyperparam_narrow_search_around_wide_55.ipynb` -- grid search for Task B
1. `create_analyze_runs_hyperparam_wide_search_small_setting.ipynb` -- grid search for Task B
1. `l1-sparsity-playground-encoder-decoder.ipynb`, `l1-sparsity-playground-encoder-decoder-Copy1.ipynb` -- learning sparse model on synthetic data

#### Task C: learning abstractions from existing observation model

1. `sparse_model_from_model.py` -- code to "sparsify" an observation model
1. `loss-landscape.ipynb` -- plotting loss landscape for X/Task C
1. `sparse-causal-learner.ipynb`

#### Sparsity playgrounds

1. `feature-selection-pid-regulator.ipynb` -- trying to adaptively adjust parameters for sparsity constraint based on the loss (does not work yet)
1. `vec-sparsity-playground.ipynb` -- applying various sparsity methods
1. `l1-sparsity-playground.ipynb`
1. `l1_constraint_tf_projected_adam.ipynb` -- implementing l1 constraint via projection


#### Baselines

1. `vec_pca_test.ipynb` -- trying to run PCA on encoded features as a baseline (negative result)
1. `environment_vec_dqn-linearnet.ipynb` -- solving VectorIncrement with a linear DQN
1. `environment_vec_dqn.ipynb` -- solving VectorIncrement with a DQN
1. `environment_vec_dqn_notransform-linearnet.ipynb` -- solving VectorIncrement on *state* using linear DQN
1. `environment_vec_dqn_notransform.ipynb` -- solving VectorIncrement on *state* using DQN
1. `environment_vec_pg-linearnet-curiosity.ipynb` -- only curiosity
1. `environment_vec_pg-linearnet.ipynb` -- linear PG on VectorIncrement
1. `environment_vec_ppo-linearnet.ipynb` -- linear PPO on VectorIncrement

#### Other files:

1. `common.py` -- helper functions
1. `curiosity.py` -- defines the curiosity reward in a TF.agents environment
1. `exp_poc.py` -- runs experiment on RL data with learning model directly from observations (does not work)
1. `sacred_search_helper.py` -- helper function running Sacred experiment for Task A
1. `vectorincrement.py` -- defines the VectorIncrement environment
1. `permutation_matrix_test.ipynb` -- which matrices are invariant under our basis change (only identity)



# Disentanglement in KeyChest environment
Another idea: Physics simulator with mechanics (blocks) from images and rotating bodies (&their images)
