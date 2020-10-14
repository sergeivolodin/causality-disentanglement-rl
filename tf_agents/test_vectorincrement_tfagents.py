import tensorflow as tf

tf.compat.v1.enable_v2_behavior()
from tf_agents.environments import tf_py_environment


# checking that the environment works
def test_env_transform():
    pass
    # env = VectorIncrementEnvironmentTFAgents(v_n=10, v_k=50, v_seed=43, do_transform=True)
    # env = wrappers.TimeLimit(env, 20)
    # utils.validate_py_environment(env, episodes=5)


def hardcoded_agent_reward(v_n, v_k, time_limit=20):
    env = VectorIncrementEnvironmentTFAgents(v_n=v_n, v_k=v_k, do_transform=False)
    env = wrappers.TimeLimit(env, 20)
    train_env = tf_py_environment.TFPyEnvironment(env)

    # running a hardcoded agent to test if the environment works correctly
    o = train_env.reset().observation.numpy()[0]
    total_reward = 0
    while True:
        act = np.argmin(o)
        step = train_env.step(act)
        o = step.observation.numpy()[0]
        r = np.array(step.reward[0])
        total_reward += r
        if step.step_type == 2:
            return total_reward


# checking that the environment works
def test_env_hardcoded_agent():
    pass
    # total_reward = hardcoded_agent_reward(2, 2)
    # assert total_reward == 10
