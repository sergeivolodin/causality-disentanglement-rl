class RandomSophisticatedFunction(object):
    """A function converting an input into a high-dimensional object."""
    def __init__(self, n=10, k=100, seed=11):

        tf.random.set_seed(seed)

        self.model = tf.keras.Sequential([
            #tf.keras.layers.Dense(10, input_shape=(n,), activation='relu'),
            #tf.keras.layers.Dense(100),
            tf.keras.layers.Dense(k, use_bias=False, kernel_initializer='random_normal'),
        ])

    def __call__(self, x):
        return self.model(np.array([x], dtype=np.float32)).numpy()[0]

assert RandomSophisticatedFunction(n=3, k=5, seed=1)([10,10,10]).shape == (5,)

class VectorIncrementEnvironment(object):
    """VectorIncrement environment."""
    def __init__(self, n=10, k=20, do_transform=True, seed=None):
        """Initialize.

        Args:
            n: state dimensionality
            k: observation dimensionality
            do_transform: if False, observation=state, otherwise observation=
              RandomSophisticatedFunction(state) with dimension k
        """
        self.n = n
        self.k = k
        self.e = RandomSophisticatedFunction(n=n, k=k, seed=seed)
        self.s = np.zeros(self.n)
        self.do_transform = do_transform

    def encoded_state(self):
        """Give the current observation."""
        if self.do_transform:
            return np.array(self.e(self.s), dtype=np.float32)
        else:
            return np.array(self.s, dtype=np.float32)  # disabling the encoding completely

    def reset(self):
        """Go back to the start."""
        self.s = np.zeros(self.n)
        return self.encoded_state()

    def step(self, action):
        """Execute an action."""
        # sanity check
        assert action in range(0, self.n)

        # past state
        s_old = np.copy(self.s)

        # incrementing the state variable
        self.s[action] += 1

        # difference between max and min entries
        # of the past state. always >= 0
        maxtomin = max(s_old) - min(s_old)

        # means that there are two different components
        if maxtomin > 0:
            # reward is proportional to the difference between the selected component
            # and the worse component to choose
            r = (max(s_old) - s_old[action]) / maxtomin
        else:
            r = 0

        return {'reward': float(r),
               'state': np.copy(self.s), # new state (do not give to the agent!)
               'observation': self.encoded_state()} # observation (give to the agent)

    def __repr__(self):
        return "VectorIncrement(state=%s, observation=%s)" % (str(self.s), str(self.encoded_state()))

