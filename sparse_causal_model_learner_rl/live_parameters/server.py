import logging
import ray


@ray.remote
class ParameterCommunicator(object):
    def __init__(self):
        self.current_parameters = {}
        self.updates = []
        self.messages = []
        self.gin_queries = []

    def gin_query(self, param):
        self.gin_queries.append(param)

    def get_clean_gin_queries(self):
        result = self.gin_queries
        self.gin_queries = []
        return result

    def add_msg(self, msg):
        self.messages.append(msg)

    def get_clear_msgs(self):
        result = self.messages
        self.messages = []
        return result

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