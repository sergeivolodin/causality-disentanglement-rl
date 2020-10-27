import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "causality_disentanglement_rl",
    version = "0.0.1",
    author = "Sergei Volodin",
    author_email = "etoestja1@gmail.com",
    description = ("Keys and Chests environment, Vector Increment env, causal reinforcement learning research code"),
    license = "BSD",
    keywords = "reinforcement learning causality gridworld gym",
    url = "https://causalrlworkshop.github.io/program/cldm_8.html",
    packages=['keychest', 'vectorincrement', 'sparse_causal_model_learner_rl', 'causal_util', 'gin_tune'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)

