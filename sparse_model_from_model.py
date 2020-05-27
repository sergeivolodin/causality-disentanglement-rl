import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

import tensorflow as tf
import numpy as np

import plotly
import plotly.graph_objs as go

from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import animation

from tqdm.notebook import tqdm

from sklearn.decomposition import PCA
from IPython.display import HTML


class LinearStateTransitionModel(object):
    """State transition model with linear dynamics."""
    def __init__(self, o, a):
        self.o = o
        self.a = a
        self.model = self.build_model()
        self.losses = []

    def build_model(self):
        """Build a model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.o, input_shape=(self.o + self.a,),
                                  activation=None, use_bias=False)
        ])

        model.compile(loss='mae', optimizer='adam')

        return model

    def fit(self, xs, ys, epochs=20, batch_size=128):
        """Fit on data."""
        assert xs.shape[1] == self.o + self.a
        assert ys.shape[1] == self.o

        history = self.model.fit(xs, ys, epochs=epochs, verbose=0,
                                 batch_size=batch_size)

        self.losses += history.history['loss']

        return history.history['loss'][-1]

    def get_Wo_Wa(self):
        """Return learned transition matrices."""
        Woa = self.model.weights[0].numpy().T
        Wo = Woa[:, :self.o]
        Wa = Woa[:, self.o:]
        return Wo, Wa

    def plot_loss(self):
        plt.plot(self.losses)


def arr_of_dicts_to_dict_of_arrays(arr):
    """ Array of dicts to dict of arrays """
    all_keys = arr[0].keys()
    return {key: [v[key] for v in arr] for key in all_keys}

def component_diff_normalized(v):
    """How much the vector is close to (1,0) or (0,1)."""
    v = np.abs(v)
    v = np.sort(v)[::-1]
    maxv = v[0]
    smaxv = v[1]
    return 1. - (maxv - smaxv) / maxv

def vec_angle_cos(v1, v2):
    """Cos betweeen vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

def repr_quality(A):
    """Loss for representation quality for matrix A."""
    columns = A.T # basis vectors = columns
    result = 0
    for i1, c1 in enumerate(columns):
        for i2, c2 in enumerate(columns):
            if i1 > i2: continue
            elif i1 == i2:
                result += component_diff_normalized(c1)
            else:
                result += np.abs(vec_angle_normalized(c1, c2))
    return result

class SparseModelLearner(object):
    """Obtain a sparser model from a given one."""
    def __init__(self, o, a, f, eps_dinv=.1, p_ord=1.,
                 optimizer=tf.keras.optimizers.Adam(lr=5e-3)):

        # obtaining the shape
        Wo = np.zeros((o, o))
        Wa = np.zeros((o, a))
        self.o, self.a = Wa.shape
        self.f = f
        assert Wo.shape == (self.o, self.o), "Wrong Wo [oxo] / Wa [oxa] shape"
        assert self.o == self.f and self.o == self.a, "Only support square matrices now..."
        self.N = self.o
        self.p_ord = p_ord

        # decoder
        self.D = tf.Variable(np.eye(self.N) + np.random.randn(self.N, self.N) * 1e-2)
        self.set_WoWa(Wo, Wa)

        # training history
        self.losses = []
        self.Ds = []
        self.eps_dinv = eps_dinv

        self.optimizer = optimizer

    def set_WoWa(self, Wo, Wa):
        """Set the input -- source transition matrix."""
        self.Wo = Wo
        self.Wa = Wa

    @property
    def Mf(self):
        return self.get_MfMa(self.D)[0].numpy()

    @property
    def Ma(self):
        return self.get_MfMa(self.D)[1].numpy()

    def get_MfMa(self, D):
        """Get M_f, M_a from decoder."""
        Mf = self.D @ self.Wo @ tf.linalg.pinv(self.D)
        Ma = self.D @ self.Wa
        return Mf, Ma

    def Mflat(self, D):
        """Get model from decoder, flat."""
        Mf, Ma = self.get_MfMa(D)
        Mflat = tf.reshape(tf.concat([Mf, Ma], axis=0), (-1,))
        return Mflat

    def plot_heatmap(self, D):
        """Plot weights heatmap."""
        plt.title("Weights heatmap")
        cm = sns.diverging_palette(0, 129, l=70, s=100, n=20, center="dark")
        Mf, Ma = self.get_MfMa(D)
        sns.heatmap(np.concatenate([Mf, Ma], axis=0), vmin=-1, vmax=1, cmap=cm)

    def process_results(self):
        """Process experimental results."""
        # Plotting everything
        fig = plt.figure(figsize=(12, 5))
        fig.patch.set_facecolor('lightgreen')

        losses = arr_of_dicts_to_dict_of_arrays(self.losses)

        N_plots = len(losses) + 1

        # model losses
        for i, key in enumerate(losses.keys()):
            plt.subplot(1, N_plots, 1 + i)
            plt.title(key)
            plt.plot(losses[key])
            plt.axhline(0, ls='--', c='black')

        plt.subplot(1, N_plots, N_plots)
        self.plot_heatmap(self.D)

        plt.show()

    def step(self):
        """One optimizer step."""
        # computing gradient
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.D)
            v = self.Mflat(self.D)
            loss_l1 = tf.linalg.norm(v, ord=self.p_ord)
            nnz = np.sum(np.abs(v.numpy()) >= 1e-2)
            loss_dinv = tf.linalg.norm(tf.reshape(tf.linalg.pinv(self.D),
                                                  (-1,)), ord=1)
            loss = loss_l1 + loss_dinv * self.eps_dinv

        grads = tape.gradient(loss, self.D)

        g1 = tape.gradient(loss_l1, self.D)
        g2 = tape.gradient(loss_dinv, self.D)

        g1 = tf.reshape(g1, (-1,)).numpy()
        g2 = tf.reshape(g2, (-1,)).numpy()
        cos = vec_angle_cos(g1, g2)

        grad_norm = tf.linalg.norm(tf.reshape(grads, (-1,)))
        self.optimizer.apply_gradients(zip([grads], [self.D]))
        result = {'loss_l1': loss_l1.numpy(), 'grad_norm': grad_norm.numpy(),
                'loss_dinv': loss_dinv.numpy(), 'loss_total': loss.numpy(),
                'cos': cos, 'nnz': nnz}
        self.losses.append(result)

    def fit(self, epochs=5000, **kwargs):
        """Fit the new matrix."""
        with tqdm(total=epochs) as pbar:
            for epoch in range(epochs):
                self.Ds.append(self.D.numpy())
                self.step(**kwargs)
                loss = self.losses[-1]
                pbar.update(1)
                pbar.set_postfix(**loss)

    def animate_weights(self, skip=50):
        """Animate weight evolution."""
        ims = []
        fig, ax = plt.subplots()
        line2d, = ax.plot([], [], lw=2)

        ax_global = ax

        def animate(i):
            #for i, d in enumerate(tqdm(Ds[::50])):
            d = self.Ds[::skip][i]
            Mf = d @ self.Wo @ np.linalg.pinv(d)
            Ma = d @ self.Wa
            Mfa = np.concatenate([Mf, Ma], axis=0)

            cm = sns.diverging_palette(0, 129, l=70, s=100, n=20, center="dark")
            im = sns.heatmap(Mfa, vmin=-1, vmax=1, cmap=cm, animated=True,
                             ax=ax_global, cbar=False)

        ani = animation.FuncAnimation(fig, animate, frames=len(self.Ds[::skip]))

        return HTML(ani.to_jshtml())

    def weights_descent_pca_space(self, maxL=10000):
        """Plot descent curve in PCA space."""
        Dflats = []
        losses_ = []

        def loss_for_d(d):
            """Loss for decoder."""
            Mf = d @ self.Wo @ np.linalg.pinv(d)
            Ma = d @ self.Wa
            Mfa = np.concatenate([Mf, Ma], axis=0)
            Mfaflat = Mfa.flatten()

            loss_l1 = np.linalg.norm(Mfaflat, ord=self.p_ord)
            loss_dinv = np.linalg.norm(np.reshape(tf.linalg.pinv(d),
                                                  (-1,)), ord=1)

            loss = loss_l1 + loss_dinv * self.eps_dinv
            loss = min(maxL, loss)
            return loss

        # computing the descent curve
        for i, d in enumerate((self.Ds)):
            Dflats.append(d.flatten())
            losses_.append(loss_for_d(d))


        # doing the PCA
        pca = PCA(n_components=2)
        Dflats_pca = pca.fit_transform(Dflats)

        # computing the mesh
        xs = np.linspace(-2, 2, 50) * 10
        ys = np.linspace(-2, 2, 50) * 10
        xys = []
        X = []
        Y = []
        for x in xs:
            for y in ys:
                xys.append((x, y))
                X.append(x)
                Y.append(y)

        # computing values on the mesh
        losses__ = []
        for d in pca.inverse_transform(xys):
            d = d.reshape(self.N, self.N)
            losses__.append(loss_for_d(d))

        # Doing 3d plot

        Z = losses__

        # Configure the trace.
        trace1 = go.Mesh3d(x=X, y=Y, z=Z, opacity=0.5)

        trace2 = go.Scatter3d(x=Dflats_pca[:, 0], y=Dflats_pca[:, 1], z=losses_,
            marker=dict(size=4, color=losses_, colorscale='Viridis',),
            line=dict(color='darkblue', width=2)
        )

        # Configure the layout.
        layout = go.Layout(
            margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
        )

        data = [trace1, trace2]

        plot_figure = go.Figure(data=data, layout=layout)

        plot_figure.update_layout(scene = dict(xaxis_title='PCA1',
                            yaxis_title='PCA2',
                            zaxis_title='loss',
                                              ),
                            width=700,
                            margin=dict(r=20, b=10, l=10, t=10))


        # Render the plot.
        return plotly.offline.iplot(plot_figure)
