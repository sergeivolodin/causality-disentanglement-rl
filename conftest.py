import gin
import pytest


@pytest.fixture(autouse=True)
def do_something():
    gin.clear_config()
