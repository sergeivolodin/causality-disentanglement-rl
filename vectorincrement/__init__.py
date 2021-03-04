import gym

from vectorincrement.vectorincrementenv import VectorIncrementEnvironment
from vectorincrement.line import LineEnvironment
from vectorincrement.sparsematrix import SparseMatrixEnvironment


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
