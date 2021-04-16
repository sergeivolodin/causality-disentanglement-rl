import gym

from vectorincrement.vectorincrementenv import VectorIncrementEnvironment
from vectorincrement.line import LineEnvironment
from vectorincrement.sparsematrix import SparseMatrixEnvironment
from vectorincrement.bounce import BounceEnv
from vectorincrement.gridworld import GridWorldNavigationEnv
from vectorincrement.epicycles import RocketEpicycleEnvironment


gym.envs.register(
    id='VectorIncrement-v0',
    entry_point=VectorIncrementEnvironment,
)

gym.envs.register(
    id='Line-v0',
    entry_point=LineEnvironment,
)

gym.envs.register(
    id='SparseMatrix-v0',
    entry_point=SparseMatrixEnvironment,
)

gym.envs.register(
    id='Bounce-v0',
    entry_point=BounceEnv,
)

gym.envs.register(
    id='GridWorldNavigation-v0',
    entry_point=GridWorldNavigationEnv
)

gym.envs.register(
    id='Epicycles-v0',
    entry_point=RocketEpicycleEnvironment
)
