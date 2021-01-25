import logging
import ray


@ray.remote
class ParameterCommunicator(object):
    def __init__(self):
        self.current_parameters = {}
        self.updates = []

    def set_current_parameters(self, current_parameters):
        self.current_parameters = current_parameters

    def get_current_parameters(self):
        return self.current_parameters

    def update_parameter(self, k, v):
        self.updates.append([k, v])

    def get_clear_updates(self):
        result = self.updates
        self.updates = []
        return result


def run_communicator(name):
    """Run the communicator which will set parameters."""
    # starting the actor
    communicator = ParameterCommunicator.options(name=name).remote()
    return communicator