from collections import OrderedDict

import gin
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import torch
from graphviz import Digraph
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import traceback
from causal_util.helpers import lstdct2dctlst
from sparse_causal_model_learner_rl.trainable.helpers import params_shape, flatten_params
from sparse_causal_model_learner_rl.trainable.helpers import unflatten_params


def plot_model(model):
    """Plot models (action and features) as a heatmap."""
    cm = sns.diverging_palette(0, 129, l=70, s=100, n=500, center="dark")

    fig = plt.figure(figsize=(10, 5))
    fig.patch.set_facecolor('xkcd:mint green')
    Mf, Ma = model.Mf, model.Ma
    plt.subplot(1, 2, 1)
    plt.title("Model for features")
    max_f = np.max(np.abs(Mf))
    sns.heatmap(Mf, vmin=-max_f, vmax=max_f, cmap=cm)
    plt.xlabel('Old features')
    plt.ylabel('New features')

    plt.subplot(1, 2, 2)
    plt.title("Model for actions")
    max_a = np.max(np.abs(Ma))
    sns.heatmap(Ma, vmin=-max_a, vmax=max_a, cmap=cm)
    plt.xlabel('Actions')
    plt.ylabel('New features')
    return fig


def select_threshold(array, name='exp', eps=1e-10, do_plot=True, do_log=True):
    """Select threshold for a matrix."""
    try:
        if not do_log:
            eps = 0
        array = np.array(array)
        # log would not work for low values
        array[array == 0.0] = eps
        aflat = np.abs(array.flatten())
        if do_log:
            aflat = np.log(aflat)
        x = pd.DataFrame({'x': aflat})
        kmeans = KMeans(n_clusters=2)
        kmeans.fit_transform(X=np.array(x.x).reshape((-1, 1)))
        x['label'] = kmeans.labels_
        clusters = np.argsort([np.min(df.x) for l, df in x.groupby('label')])
        l = np.max(x.x[x.label == clusters[0]])
        r = np.min(x.x[x.label == clusters[1]])
        assert l < r
        threshold = (l + r) / 2

        if do_plot:
            fig = plt.figure()
            plt.hist(x.x)
            plt.axvline(threshold, label='threshold')
            plt.legend()
            plt.savefig(f"threshold_{name}.png", bbox_inches='tight')
            plt.clf()
            plt.close(fig)
        res = threshold
        if do_log:
            threshold = np.exp(threshold)
        return threshold
    except Exception as e:
        if np.isnan(array).any():
            raise ValueError(f"Threshold selection failed (NaN): {name} {type(e)} {e} {array}")
        else:
            print(f"Threshold selection failed (no NaN): {name} {type(e)} {e} {array}")
            print(traceback.format_exc())
            return 0.0


def graph_for_matrices(model, threshold_act=0.2, threshold_f=0.2, do_write=True):
    """Visualize matrices as a graph."""
    Mf, Ma = model.Mf, model.Ma
    # dimension
    actions = Ma.shape[1]
    features = Mf.shape[1]

    ps = Digraph(name='Causal model', engine='neato')  # ,
    # node_attr={'shape': 'plaintext'})

    # adding features nodes
    for f in range(features):
        ps.node('f%02d' % f, color='green')
        ps.node("f'%02d" % f, color='blue')

    # adding action edges
    for a in range(actions):
        ps.node('a%02d' % a, color='red')

    # adding edges
    edges = 0

    for f1, a in zip(*np.where(np.abs(Ma) > threshold_act)):
        ps.edge('a%02d' % a, "f'%02d" % f1)
        edges += 1

    for f1, f in zip(*np.where(np.abs(Mf) > threshold_f)):
        ps.edge('f%02d' % f, "f'%02d" % f1)
        edges += 1

    max_edges = features ** 2 + actions * features
    percent = int(100 - 100. * edges / max_edges)
    # print("Number of edges: %d out of %d, sparsity %.2f%%" % \
    #       (edges, max_edges, percent))

    f_out = None
    if do_write:
        f_out = f"CausalModel"
        ps.render(filename=f_out, format='png')

    return ps, f_out


def get_weights_from_learner(learner, weight_names):
    """Get history from a learner for specific weights only."""

    keys = [f"weights/{weight}" for weight in weight_names]
    history = lstdct2dctlst(learner.history)
    lengths = [len(history[key]) for key in keys]
    assert all(lengths[0] == l for l in lengths)
    result = []
    for i in range(lengths[0]):
        weights_now = [history[f"weights/{weight}"][i] for weight in weight_names]
        result.append(weights_now)

    return result


def total_loss(learner, opt_label='opt1'):
    """Get total loss for an optimizer"""
    total_loss = 0
    for loss_label in learner.config['execution'][opt_label]:
        loss = learner.config['losses'][loss_label]
        value = loss['fcn'](**learner._context)
        coeff = loss['coeff']
        total_loss += coeff * value

    return total_loss.item() if hasattr(total_loss, 'item') else total_loss


def set_weights(weights, data_numpy):
    """Set weights from numpy arrays."""
    assert len(weights) == len(data_numpy)
    for w, data in zip(weights, data_numpy):
        w.data = torch.from_numpy(data).to(w.dtype).to(w.device)


def with_weights(weights_list, dest_shape=None):
    """Decorate a function: make it take additional weights argument."""

    def wrap(f):
        def g(w, weights_list=weights_list, dest_shape=dest_shape, *args, **kwargs):
            """Call f with given weights."""

            # unflattening paramters if requested
            if dest_shape is not None:
                w = unflatten_params(w, dest_shape)

            # setting weights
            set_weights(weights=weights_list, data_numpy=w)

            # calling the original function
            return f(*args, **kwargs)

        return g

    return wrap


def weight_name_to_param(trainables, name):
    """Return a torch variable corresponding to a name in trainables."""
    trainable_name, weight_name = name.split('/')
    return OrderedDict(trainables[trainable_name].named_parameters())[weight_name]


def select_weights(trainables, weight_names):
    """Select weights from models by names."""
    return [weight_name_to_param(trainables, w) for w in weight_names]


def loss_and_history(learner, loss, weight_names):
    """Return loss function and flat history, given weight names."""
    # list of ALL trainable variables
    trainables = learner.trainables

    # relevant weights history
    weights_history = get_weights_from_learner(learner, weight_names)

    # parameters to track/vary
    weights = select_weights(trainables, weight_names)

    # destination shape
    shape = params_shape(weights)

    # function taking parameters and outputting loss
    loss_w = with_weights(weights, dest_shape=shape)(loss)

    # history of weight changes (flattened)
    flat_history = [flatten_params(p) for p in weights_history]

    return loss_w, flat_history


@gin.configurable
def plot_contour(flat_history, loss_w, scale=5, n=50):
    """Contour plot from PCA history with loss values."""
    pca = PCA(n_components=2)
    flat_history_pca = pca.fit_transform(flat_history)

    R = np.max(np.abs(flat_history_pca), axis=0)
    R *= scale

    x = np.linspace(-R[0], R[0], n)
    y = np.linspace(-R[1], R[1], n)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            xys = np.array([[X[i, j], Y[i, j]]])
            w = pca.inverse_transform(xys)[0]
            Z[i, j] = loss_w(w)

    fig, ax = plt.subplots(figsize=(10, 20))
    ax.set_title('Loss contour plot')

    Zlog = np.log(Z)
    extent = (-R[0], R[0], -R[1], R[1])

    im = ax.imshow(Zlog, interpolation='bilinear', origin='lower',
                   cmap=cm.RdGy, extent=extent)

    levels = np.linspace(np.min(Zlog), np.max(Zlog), 10)

    CS = ax.contour(Zlog, levels, origin='lower', extend='both',
                    cmap='gray',
                    linewidths=2, extent=extent)

    # make a colorbar for the contour lines
    # CB = fig.colorbar(CS, shrink=0.8)

    ax.clabel(CS, inline=True, fontsize=10)

    # We can still add a colorbar for the image, too.
    CBI = fig.colorbar(im, orientation='horizontal', shrink=0.8)

    # l, b, w, h = ax.get_position().bounds
    # ll, bb, ww, hh = CB.ax.get_position().bounds
    # CB.ax.set_position([ll, b + 0.1*h, ww, h*0.8])

    plt.plot(*zip(*flat_history_pca))
    plt.scatter(*flat_history_pca[0], s=200, marker='<', color='blue', label='Start')
    plt.scatter(*flat_history_pca[-1], s=200, marker='*', color='blue', label='End')

    plt.legend()

    return fig, ax


def get_mesh(scale=5, n=50):
    """Get a mesh of a given scale with a given number of points."""
    # computing the mesh
    xs = np.linspace(-scale, scale, n)
    ys = np.linspace(-scale, scale, n)

    xys = []
    X = []
    Y = []
    for x in xs:
        for y in ys:
            xys.append((x, y))
            X.append(x)
            Y.append(y)
    return xs, ys, xys, X, Y


@gin.configurable
def plot_3d(flat_history, loss_w, scale=5, n=50):
    """Plot the 3D loss landscape."""

    pca = PCA(n_components=2)
    flat_history_pca = pca.fit_transform(flat_history)

    losses = [loss_w(w) for w in flat_history]

    z_step_fraction = 0.1

    R = np.max(np.linalg.norm(flat_history_pca, axis=1))
    R *= scale
    xs, ys, xys, X, Y = get_mesh(n=n, scale=R)

    # computing values on the mesh
    losses_mesh = []
    for params in pca.inverse_transform(xys):
        losses_mesh.append(loss_w(params))

    Z = losses_mesh

    Zmin = np.min(Z)
    Zmax = np.max(Z)
    Zstep = (Zmax - Zmin) * z_step_fraction

    # Doing 3d plot

    lighting = dict(ambient=0.4,
                    diffuse=1,
                    fresnel=4,
                    specular=0.5,
                    roughness=0.05)

    lightposition = dict(x=0,
                         y=5,
                         z=min(10000, Zmax + 5))

    trace2 = go.Scatter3d(x=flat_history_pca[:, 0], y=flat_history_pca[:, 1], z=losses,
                          marker=dict(size=4, color=losses, ),
                          line=dict(color='darkblue', width=2)
                          )

    trace3 = go.Surface(x=xs, y=ys, z=np.array(Z).reshape(n, n).T, opacity=0.5,
                        contours_z=dict(show=True, usecolormap=True,
                                        highlightcolor="limegreen", project_z=True),
                        lighting=lighting,
                        lightposition=lightposition

                        )

    # Configure the layout.
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )

    data = [trace2, trace3]

    plot_figure = go.Figure(data=data, layout=layout)

    plot_figure.update_layout(scene=
                              dict(xaxis_title='PCA1',
                                   yaxis_title='PCA2',
                                   zaxis_title='loss'),
                              width=700,
                              margin=dict(r=20, b=10, l=10, t=10))

    return plot_figure
