class ComplexityMetric(object):
    def __init__(self, **kwargs):
        self.params = kwargs

    def __call__(self, obj):
        return NotImplementedError
