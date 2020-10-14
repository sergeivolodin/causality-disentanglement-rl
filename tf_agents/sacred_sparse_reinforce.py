import os
import numpy as np

# randomly selecting a GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % np.random.randint(0, 2)

# always only using the memory we need, and not more
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

import tensorflow as tf

tf.compat.v1.enable_v2_behavior()

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.networks import actor_distribution_network
from tf_agents.utils import common
from tf_agents.environments import wrappers

from tf_agents.common import *
from tf_agents.curiosity import *
from tf_agents.sparse_model_from_model import LinearStateTransitionModel, SparseModelLearner

from tqdm.notebook import tqdm

from sacred import Experiment
from sacred.observers import MongoObserver

import pickle
import uuid

ex = Experiment("sparse_causal_model_rl", interactive=True)
ex.observers.append(MongoObserver(url='127.0.0.1:27017',
                                  db_name='test'))

#@ex.config
def exp_config_small():
    """All configuration."""
    ### Environment parameters
    v_n = 5
    v_k = 5
    v_seed = 43
    do_transform = True
    time_limit = 20
    
    
    ### Agent hyperparameters
    num_iterations = 10 # @param {type:"integer"}
    collect_episodes_per_iteration = 2 # @param {type:"integer"}
    replay_buffer_capacity = 1000 # @param {type:"integer"}

    fc_layer_params = ()

    learning_rate = 1e-3 # @param {type:"number"}
    log_interval = 1 # @param {type:"integer"}
    num_eval_episodes = 10 # @param {type:"integer"}
    eval_interval = 1 # @param {type:"integer"}

    # p norm
    p_ord = 1

    # regularization for reconstruction
    eps_dinv = 1.

    d_init_randomness = 5.

    # for training observation model
    model_W_train_epochs = 50

    # for training feature model
    model_sml_train_epochs = 50

    ### Curiosity parameters
    # curiosity reward coefficient
    alpha = 1.0

    # how often to run curiosity/model training?
    curiosity_interval = 1
    
    # maximal number of data points for W dataset
    w_max_dataset_size = 1000

@ex.config
def exp_config_big():
    """All configuration."""
    ### Environment parameters
    v_n = 5
    v_k = 5
    v_seed = 43
    do_transform = True
    time_limit = 20
    
    
    ### Agent hyperparameters
    num_iterations = 1000 # @param {type:"integer"}
    collect_episodes_per_iteration = 20 # @param {type:"integer"}
    replay_buffer_capacity = 1000 # @param {type:"integer"}

    fc_layer_params = ()

    learning_rate = 1e-3 # @param {type:"number"}
    log_interval = 25 # @param {type:"integer"}
    num_eval_episodes = 10 # @param {type:"integer"}
    eval_interval = 1 # @param {type:"integer"}

    # p norm
    p_ord = 1

    # regularization for reconstruction
    eps_dinv = 1.

    d_init_randomness = 5.

    # for training observation model
    model_W_train_epochs = 500

    # for training feature model
    model_sml_train_epochs = 1000

    ### Curiosity parameters
    # curiosity reward coefficient
    alpha = 1.0

    # how often to run curiosity/model training?
    curiosity_interval = 10
    
    # maximal number of data points for W dataset
    w_max_dataset_size = 5000

@ex.capture
def get_env(sml, v_n, v_k, v_seed, do_transform, alpha, time_limit,
            add_curiosity_reward=True):
    """Return a copy of the environment."""
    env = VectorIncrementEnvironmentTFAgents(v_n=v_n, v_k=v_k, v_seed=v_seed,
                                             do_transform=do_transform)
    env = wrappers.TimeLimit(env, time_limit)
    if add_curiosity_reward:
        env = CuriosityWrapper(env, sml.env_model, alpha=alpha)
    env = tf_py_environment.TFPyEnvironment(env)
    return env

@ex.capture
def get_W_sml(v_n, v_k, p_ord, eps_dinv, d_init_randomness):
    """Get model learners."""
    W = LinearStateTransitionModel(o=v_k, a=v_n)
    sml = SparseModelLearner(o=v_k, a=v_n, f=v_n, p_ord=p_ord,
                             eps_dinv=eps_dinv, d_init_randomness=d_init_randomness)
    return W, sml

@ex.capture
def get_envs(sml):
    """Get environments."""
    train_env = get_env(sml=sml, add_curiosity_reward=True)
    eval_env = get_env(sml=sml, add_curiosity_reward=False)
    return train_env, eval_env

@ex.capture
def get_reinforce_agent(train_env, fc_layer_params, v_n, v_k, learning_rate):
    """Get REINFORCE tf.agent."""
    decoder_layer_agent = tf.keras.layers.Dense(v_n, input_shape=(v_k,), activation=None,
                             use_bias=False, kernel_initializer='random_normal')

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params,
        activation_fn=tf.keras.activations.relu,
        preprocessing_layers=decoder_layer_agent
        # for features: add preprocessing_layers=[...]
    )

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.compat.v2.Variable(0)

    tf_agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter)
    tf_agent.initialize()

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    return tf_agent, actor_net, optimizer

@ex.capture
def train_agent(tf_agent, train_env, eval_env, num_iterations, num_eval_episodes, collect_episodes_per_iteration,
                v_n, model_W_train_epochs, model_sml_train_epochs, replay_buffer_capacity, optimizer,
                eval_interval, curiosity_interval, W, sml, actor_net, _run, w_max_dataset_size):
    """Train a tf.agent with sparse model."""
    
    decoder_layer_agent = actor_net.layers[0].layers[0] # taking the copied layer with actual weights

    # sml initializes D properly
    decoder_layer_agent.set_weights([sml.D.numpy().T])

    
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)

    curiosity_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=1000000) # should never overflow
    
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    tf_agent.train = common.function(tf_agent.train)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    train_avg_return = compute_avg_return(eval_env, tf_agent.collect_policy, num_eval_episodes)
    returns = [avg_return]
    train_returns = [train_avg_return]

    with tqdm(total=num_iterations, desc="Iterations") as pbar:
        for iteration in range(num_iterations):

          # Collect a few episodes using collect_policy and save to the replay buffer.
          collect_episode(train_env, tf_agent.collect_policy,
                          collect_episodes_per_iteration,
                          [replay_buffer, curiosity_replay_buffer])

          # Use data from the buffer and update the agent's network.
          experience = replay_buffer.gather_all()
          train_loss = tf_agent.train(experience)
          replay_buffer.clear()

          #print("Agent train step")

          step = tf_agent.train_step_counter.numpy()
            
          _run.log_scalar("agent.train_loss", train_loss.loss.numpy(), iteration)

          if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            train_avg_return = compute_avg_return(train_env, tf_agent.collect_policy, num_eval_episodes)
            pbar.set_postfix(train_c_return=train_avg_return,
                             eval_return=avg_return,
                             agent_loss=train_loss.loss.numpy())
            returns.append(avg_return)
            train_returns.append(train_avg_return)
            _run.log_scalar("agent.eval_return", avg_return, iteration)
            _run.log_scalar("agent.train_return", train_avg_return, iteration)

          if step % curiosity_interval == 0 and len:
            #clear_output()
            xs, ys = buffer_to_dataset(curiosity_replay_buffer, v_n)
            
            _run.log_scalar("causal.total_dataset_size", len(xs), iteration)
            
            # prevent the dataset from growing in an unbounded way
            if len(xs) > w_max_dataset_size:
                idxes = np.random.choice(range(len(xs)), w_max_dataset_size, replace=False)
                xs = np.array(xs)[idxes]
                ys = np.array(ys)[idxes]
            
            _run.log_scalar("causal.used_dataset_size", len(xs), iteration)

            # fitting on observational data...
            losses = W.fit(xs=xs, ys=ys, epochs=model_W_train_epochs)
            for l in losses:
                _run.log_scalar("W.fit", l)
            #W.plot_loss()

            # setting weights from the agent to the model...
            sml.D.assign(decoder_layer_agent.get_weights()[0].T)

            # setting the new observation transition matrix
            sml.set_WoWa(*W.get_Wo_Wa())

            def sml_callback(loss):
                for k, v in loss.items():
                    _run.log_scalar("sml.%s" % k, v)
            
            # fitting the SML model
            sml.fit(epochs=model_sml_train_epochs, loss_callback=sml_callback)

            # setting weights from the model to the agent...
            decoder_layer_agent.set_weights([sml.D.numpy().T])

            #agent_replay_buffer.clear()
            # observations are actually the same


            #print("Model train step")
          pbar.update(1)
        
    return returns, train_returns

@ex.automain
def run_agent(_run):
    global W, sml
    W, sml = get_W_sml()
    train_env, eval_env = get_envs(sml=sml)
    tf_agent, actor_net, optimizer =\
        get_reinforce_agent(train_env=train_env)

    returns, train_returns = train_agent(tf_agent=tf_agent,
                                         train_env=train_env,
                                         eval_env=eval_env,
                                         optimizer=optimizer,
                                         W=W, sml=sml,
                                         actor_net=actor_net)
    
    fn = 'sml_%s.pkl' % str(uuid.uuid4())
    pickle.dump(sml, open(fn, 'wb'))
    
    _run.add_artifact(fn, "sml")
    
    fn = 'W_%s.pkl' % str(uuid.uuid4())
    pickle.dump(W, open(fn, 'wb'))
    
    _run.add_artifact(fn, "W")
    
    _run.log_scalar("agent.train_count",
                    tf_agent.train_step_counter.numpy())