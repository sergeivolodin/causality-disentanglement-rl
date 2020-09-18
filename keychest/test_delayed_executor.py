from keychest.keychestenv import DelayedExecutor


class ProbeFunction(object):
    """Log number of calls for all objects."""
    NUM_OBJECTS = 0
    
    def __init__(self):
        self.num_calls = 0
        self.idx = ProbeFunction.NUM_OBJECTS
        ProbeFunction.NUM_OBJECTS += 1
    def __call__(self):
        self.num_calls += 1
        print(self, "called")
        assert self.num_calls <= 1, f"Too many calls {self}"
        return self
        
    def __repr__(self):
        return f"Probe function {self.idx}, total {self.num_calls}"


def test_delayed_executor():
    de = DelayedExecutor()
    assert len(de.queue) == 0
    f0 = ProbeFunction()
    f1 = ProbeFunction()
    f2 = ProbeFunction()
    de.push(0, f0)
    de.push(1, f1)
    de.push(2, f2)
    de.step()
    assert f0.num_calls == 1
    assert f1.num_calls == 0
    assert f2.num_calls == 0
    de.step()
    assert f0.num_calls == 1
    assert f1.num_calls == 1
    assert f2.num_calls == 0
    de.step()
    assert f0.num_calls == 1
    assert f1.num_calls == 1
    assert f2.num_calls == 1

    for _ in range(10):
        de.step()

    f0 = ProbeFunction()
    f1 = ProbeFunction()
    f2 = ProbeFunction()
    de.push(0, f0)
    de.push(1, f1)
    de.push(2, f2)
    de.step()
    assert f0.num_calls == 1
    assert f1.num_calls == 0
    assert f2.num_calls == 0
    de.step()
    assert f0.num_calls == 1
    assert f1.num_calls == 1
    assert f2.num_calls == 0
    de.step()
    assert f0.num_calls == 1
    assert f1.num_calls == 1
    assert f2.num_calls == 1
