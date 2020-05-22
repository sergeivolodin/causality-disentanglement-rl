import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tqdm.notebook import tqdm

tf.compat.v1.enable_v2_behavior()

from curiosity import m_passthrough_action

import pixiedust
from functools import partial
import gin
import itertools
import multiprocessing
import pickle

from copy import deepcopy

from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks

def matrix_dist(A, B):
    return np.linalg.norm((A - B).flatten(), ord=1)

def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    if np.linalg.norm(v, ord=1) <= z:
        return v
        
    # algorithm from
    # https://gist.github.com/EdwardRaff/f4f4cf0c927c2addfb39
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

def project_l1(v, z=1):
    """Project onto |x|_1<=z."""
    v = np.array(v)
    signs = np.sign(v)
    v_unsigned = np.multiply(v, signs)
    u_unsigned = projection_simplex_sort(v_unsigned, z)
    u = np.multiply(u_unsigned, signs)
    norm = np.linalg.norm(u, ord=1)
    assert norm <= z * 1.1, "Got norm %.2f %s" % (norm, str(u))
    #assert np.allclose(np.linalg.norm(u, ord=1), 1)
    return u

@gin.configurable
def projection_step(w, l1_ball_size=None):
    """Project weights onto an l1 ball."""
    #assert len(model.trainable_variables) == 1,\
    #  "Only support 1 weight tensor (now)."
    weights = w
    #w_flat = flatten_array_of_tensors(weights)
    w_numpy = weights.numpy()
    w_flat = np.reshape(w_numpy, (-1,))
    
    if l1_ball_size is None:
        norm = np.linalg.norm(w_flat, ord=1)
        return norm
    
    w_flat_new = project_l1(w_flat, l1_ball_size)
    norm = np.linalg.norm(w_flat_new, ord=1)
    w_flat_new = np.reshape(w_flat_new, w_numpy.shape)
    weights.assign(w_flat_new)
    return norm

@gin.configurable
def mask_step(w, mask=None, eps=1e-3):
    """Project weights onto an l1 ball."""
    
    #assert len(model.trainable_variables) == 1,\
    #  "Only support 1 weight tensor (now)."
    weights = w # model.trainable_variables[0]
    nnz = np.sum(np.abs(weights.numpy()) > eps)
    
    if mask is None: return nnz
    w_numpy = weights.numpy()
    assert w_numpy.shape == mask.shape, "Bad mask shape %s %s" % \
      (str(w_numpy.shape), str(mask.shape))
    weights.assign(np.multiply(mask, w_numpy))
    nnz = np.sum(np.abs(weights.numpy()) > eps)
    return nnz

@gin.configurable
def compute_mask(w, components_to_keep=5):
    """Select biggest components in w."""
    
    # assuming no duplicates...
    
    assert isinstance(components_to_keep, int), "Want int, got %s" % \
      str(components_to_keep)
    assert components_to_keep > 0, "No sense doing 0 components"

    # weights of the model
    w_flat = w.flatten()

    # number of components in weights
    n_components = w_flat.shape[0]

    #print(n_components, components_to_keep)
    
    # clamping components_to_keep
    if components_to_keep > n_components:
        
        components_to_keep = n_components

    # selecting the threshold so that we have
    # the correct number of components
    threshold = np.sort(np.abs(w_flat))[::-1][components_to_keep - 1]
    
    # print(w_flat, threshold)
    
    # abs of w
    w_abs = np.abs(w)

    # computing the mask
    mask = 1. * (w_abs > threshold)
    left = components_to_keep - np.sum(mask > 0)
    a, b = np.where(w_abs == threshold)
    for i in range(left):
        mask[a[i], b[i]] = 1.0

    assert np.sum(mask > 0) == components_to_keep,\
      "Wrong mask %s %d" % (str(mask), components_to_keep)
    
    return mask

class MaxNorm1(tf.keras.constraints.Constraint):
  def __init__(self, max_value=2, axis=0):
    self.max_value = max_value
    self.axis = axis

  def __call__(self, w):
    norms = K.sqrt(
        math_ops.reduce_sum(math_ops.square(w), axis=self.axis, keepdims=True))
    desired = K.clip(norms, 0, self.max_value)
    return w * (desired / (K.epsilon() + norms))

  def get_config(self):
    return {'max_value': self.max_value, 'axis': self.axis}

projection_simplex_sort(np.array([1,0.1]), 3)

#%%pixie_debugger

# l1 parameters
l1coeff = 0

# for reconstructor
#l2coeff = 0

# for keras sparsity
sparsity = 0.3
pruning_params = {
        'pruning_schedule': pruning_sched.ConstantSparsity(0.9, 0),
        #'pruning_schedule': pruning_sched.PolynomialDecay(0, 0.3, 0, 100),
        'block_size': (1, 1),
        'block_pooling_type': 'AVG'
    }

def component_diff_normalized(v):
    """How much the vector is close to (1,0) or (0,1)."""
    v = np.abs(v)
    return 1. - (max(v) - min(v)) / max(v)

def vec_angle_normalized(v1, v2):
    """Cos betweeen vectors."""
    return np.abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def repr_quality(A):
    """Loss for representation quality for matrix A."""
    [s_1, s_2] = A.T # basis vectors = columns
    return component_diff_normalized(s_1) + \
           component_diff_normalized(s_2) + \
           vec_angle_normalized(s_1, s_2)

@gin.configurable
def build_decoder_model(input_layer, init_fp_dist=None):
    """Create a decoder model."""
    decoder = tf.keras.Sequential([ #D
        input_layer,
        tf.keras.layers.Dense(2, activation=None, use_bias=False, #kernel_regularizer=tf.keras.regularizers.l2(l2coeff),
                             #kernel_initializer='random_normal',
                             #kernel_constraint=tf.keras.constraints.UnitNorm()
                             kernel_constraint=tf.keras.constraints.MinMaxNorm(0.5, 1.5) # == 1 --
                             ),
    ])

    if init_fp_dist is not None:
        decoder.layers[-1].set_weights([np.linalg.inv(Q1).T +\
                                        np.ones((2, 2)) * init_fp_dist])
    return decoder

@gin.configurable
def build_reconstructor_model(init_fp_dist=None):
    """Build the reconstructor."""
    # encoder model -- imitates the RL agent which has converged to something -- and needs to reconstruct the state
    # but the policy is "fixed" and the reward = max
    reconstructor = tf.keras.Sequential([ # E
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(2, activation=None, use_bias=False, #kernel_regularizer=tf.keras.regularizers.l2(l2coeff),
                             #kernel_initializer='random_normal',
                             #kernel_constraint=tf.keras.constraints.UnitNorm()

                             # how can we take the scale out of this -- decompose
                             kernel_constraint=tf.keras.constraints.MinMaxNorm(0.5, 1.5)
                             ),
    ])

    if init_fp_dist is not None:
        reconstructor.layers[-1].set_weights([Q1.T + np.ones((2, 2)) * init_fp_dist])
        #reconstructor.layers[-1].set_weights([np.linalg.inv(decoder.get_weights()[0])])

    return reconstructor

@gin.configurable
def build_feature_model(decoder, init_fp_dist=None, l1coeff=0.0):
    """Build the feature transition dynamics model."""
    # maps observations to features
    model = tf.keras.Sequential([ # M
        m_passthrough_action(decoder, 2, 2), # D
        tf.keras.Input(shape=(4,)),
        #prune.prune_low_magnitude(
            tf.keras.layers.Dense(2, activation=None, use_bias=False, kernel_regularizer=tf.keras.regularizers.l1(l1coeff),
                                 #kernel_initializer='random_normal'
                                 ), # M_D
        #**pruning_params)
    ])

    if init_fp_dist is not None:
        model.layers[-1].set_weights([A.T + np.ones((4, 2)) * init_fp_dist])
    return model

def loss_model_fit(y_true, y_pred, decoder=None, sample_weight=None):
    """How well the model fits the data?"""
    del sample_weight
    # y_pred = from the model
    L = tf.reduce_mean(tf.abs(y_pred - decoder(y_true)))
    return L
    
def loss_model_fit_rmd(y_true, y_pred, reconstructor=None, sample_weight=None):
    """How well the model fits the data?"""
    del sample_weight
    # y_pred = from the model
    L = tf.reduce_mean(tf.abs(reconstructor(y_pred) - y_true))
    return L

def loss_reconstructor(reconstructor, decoder, x):
    """How well the reconstructor can obtain observations?"""
    # x is the input tensor (observations)
    if x is None: return 0
    x = x[:, :2]
    L = tf.reduce_mean(tf.abs(reconstructor(decoder(x)) - x))
    return L

def list_of_lists_to_list(lst_of_lst):
    """Flatten a list of lists."""
    return [x for lst in lst_of_lst for x in lst]
    
@tf.function
def flatten_array_of_tensors(W):
    """Take an array of tensor and turn into a single flat tensor."""
    return tf.concat([tf.reshape(w, (-1,)) for w in W], axis=0)

def apply_optimizer(loss, models, optimizer, tape):
    """Do a step on the loss."""
    # all their variables
    all_variables = [model.trainable_variables for model in models]

    grads = tape.gradient(loss, all_variables)
    optimizer.apply_gradients(zip(list_of_lists_to_list(grads),
                                  list_of_lists_to_list(all_variables)))


@gin.configurable
def step_rmd(model, decoder, reconstructor, xs, ys,
         optimizer,
         l_rec_coeff=1.0,
         l1_coeff=0.0):
    """One optimization step."""
    # xs - observations + actions
    # ys - next observations

    # converting dtype
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    with tf.GradientTape() as tape:
        # Make prediction
        y_pred = model(xs)

        # Calculate loss
        l_fit = loss_model_fit_rmd(ys, y_pred, reconstructor=reconstructor)
        l_rec = loss_reconstructor(reconstructor=reconstructor,
                                   decoder=decoder, x=xs)
                                                # weight 0 is decoder
        l_l1 = tf.norm(flatten_array_of_tensors([model.weights[1]]),
                       ord=1)

        # total loss
        total_loss = l_fit + l_rec_coeff * l_rec + \
                     l1_coeff * l_l1

    # list of models
    models = [model, reconstructor] # decoder weights are in the model

    apply_optimizer(loss=total_loss, optimizer=optimizer,
                    tape=tape, models=models)
            
    nnz = mask_step(model.weights[1])
    l1 = projection_step(model.weights[1])

    return {'l_fit': l_fit.numpy(), 'l_rec': l_rec.numpy(),
            'l_l1': l1, 'nnz': nnz}


@gin.configurable
def step_rmd_2opt(model, decoder, reconstructor, xs, ys,
         optimizer_rmd, optimizer_rd,
         l_rec_coeff=1.0,
         l1_coeff=0.0):
    """One optimization step."""
    # xs - observations + actions
    # ys - next observations

    # converting dtype
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    
    # left out deliberately, don't want to update on it.
    l_l1 = tf.norm(flatten_array_of_tensors([model.weights[1]]),
                   ord=1)

    with tf.GradientTape() as tape_rmd, tf.GradientTape() as tape_rd:
        # Make prediction
        y_pred = model(xs)

        # Calculate loss
        l_fit = loss_model_fit_rmd(ys, y_pred, reconstructor=reconstructor)
        l_rec = loss_reconstructor(reconstructor=reconstructor,
                                   decoder=decoder, x=xs)
                                                # weight 0 is decoder
        
        rmd_loss = l_fit
        rd_loss  = l_rec

    # list of models
    models_rmd = [model, reconstructor] # decoder weights are in the model
    models_rd  = [reconstructor, decoder]
    
    nnz = mask_step(model.weights[1])
    l1 = projection_step(model.weights[1])
    
    # returning the old loss to output post-projection values
    results = {'l_fit': l_fit.numpy(), 'l_rec': l_rec.numpy(),
            'l_l1': l1, 'nnz': nnz}

    apply_optimizer(loss=rmd_loss, optimizer=optimizer_rmd,
                    tape=tape_rmd, models=models_rmd)
    apply_optimizer(loss=rd_loss, optimizer=optimizer_rd,
                    tape=tape_rd, models=models_rd)

    return results
    
@gin.configurable
def step_2opt(model, decoder, reconstructor, xs, ys,
         optimizer_md, optimizer_rd,
         l_rec_coeff=1.0,
         l1_coeff=0.0):
    """One optimization step."""
    # xs - observations + actions
    # ys - next observations

    # converting dtype
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    
    # left out deliberately, don't want to update on it.
    l_l1 = tf.norm(flatten_array_of_tensors([model.weights[1]]),
                   ord=1)

    with tf.GradientTape() as tape_md, tf.GradientTape() as tape_rd:
        # Make prediction
        y_pred = model(xs)

        # Calculate loss
        l_fit = loss_model_fit(ys, y_pred, decoder=decoder)
        l_rec = loss_reconstructor(reconstructor=reconstructor,
                                   decoder=decoder, x=xs)
                                                # weight 0 is decoder
        
        md_loss = l_fit
        rd_loss  = l_rec

    # list of models
    models_md = [model] # decoder weights are in the model
    models_rd  = [reconstructor, decoder]
    
    # returning the old loss to output post-projection values
    results = {'l_fit': l_fit.numpy(), 'l_rec': l_rec.numpy()}

    apply_optimizer(loss=md_loss, optimizer=optimizer_md,
                    tape=tape_md, models=models_md)
    apply_optimizer(loss=rd_loss, optimizer=optimizer_rd,
                    tape=tape_rd, models=models_rd)
                    
    nnz = mask_step(model.weights[1])
    l1 = projection_step(model.weights[1])
    
    results['nnz'] = nnz
    results['l_l1'] = l1

    return results

@gin.configurable
def step(model, decoder, reconstructor, xs, ys, optimizer,
         l_rec_coeff=1.0,
         l1_coeff=0.0):
    """One optimization step."""
    # xs - observations + actions
    # ys - next observations

    # converting dtype
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    with tf.GradientTape() as tape:
        # Make prediction
        y_pred = model(xs)

        # Calculate loss
        l_fit = loss_model_fit(ys, y_pred, decoder=decoder)
        l_rec = loss_reconstructor(reconstructor=reconstructor,
                                   decoder=decoder, x=xs)
                                                # weight 0 is decoder
        l_l1 = tf.norm(flatten_array_of_tensors([model.weights[1]]),
                       ord=1)

        # total loss
        total_loss = l_fit + l_rec_coeff * l_rec + \
                     l1_coeff * l_l1

    # list of models
    models = [model, reconstructor] # decoder weights are in the model

    apply_optimizer(loss=total_loss, optimizer=optimizer,
                    tape=tape, models=models)
            
    nnz = mask_step(model.weights[1])
    l1 = projection_step(model.weights[1])

    return {'l_fit': l_fit.numpy(), 'l_rec': l_rec.numpy(),
            'l_l1': l1, 'nnz': nnz}

def arr_of_dicts_to_dict_of_arrays(arr):
    """ Array of dicts to dict of arrays """
    all_keys = arr[0].keys()
    return {key: [v[key] for v in arr] for key in all_keys}

@gin.configurable
def get_results(xs_e, ys_e, Q1, batch_size=16, epochs=1, step=None, l_rec_coeff=1):
    """Compute results."""
    # input for the decoder (features)
    inp_dec = tf.keras.Input(shape=(2,))

    # defining models
    decoder = build_decoder_model(inp_dec)
    model = build_feature_model(decoder)
    reconstructor = build_reconstructor_model()

    # compiling
    model.compile(optimizer='adam', loss=partial(loss_model_fit,
                                                decoder=decoder))

    # for weight pruning
    #step_callback = pruning_callbacks.UpdatePruningStep()
    #step_callback.set_model(model)
    #step_callback.on_train_begin()

    losses = []
    distances = []

    def evaluate_dist():
        """Evaluate the models."""
        DE = decoder.weights[0].numpy().T @ Q1
        distances.append(repr_quality(DE))

    # training mechanics
    x_train = xs_e
    y_train = ys_e
    bat_per_epoch = int(len(x_train) / batch_size)

    # evaluate the quality
    evaluate_dist()

    # epoch loop
    with tqdm(total=epochs, ncols='100%') as pbar:
      for epoch in range(epochs):
        L = []

        # one epoch
        #with tqdm(total=bat_per_epoch, leave=False) as pbar1:
        for i in range(bat_per_epoch):
            #step_callback.on_train_batch_begin(batch=-1)
            n = i * batch_size
            loss = step(model=model, decoder=decoder,
                          reconstructor=reconstructor,
                          xs=x_train[n:n+batch_size],
                          ys=y_train[n:n+batch_size])
            L.append(loss)

            # hard sparsity
            #l = model.layers[-1]
            #w = l.get_weights()[0]
            #w[np.abs(w) < 1e-1] = 0
            #l.set_weights([w])

            # keras sparsity
            #step_callback.on_train_batch_end(batch=i)
            #pbar1.update(1)

        # keras sparsity
        #step_callback.on_epoch_end(batch=-1)
        #step_callback.on_epoch_end(-1)
        #print(model.get_weights())

        # adding metrics
        L = arr_of_dicts_to_dict_of_arrays(L)
        curr_loss = {x: np.mean(y) for x, y in L.items()}
        losses.append(curr_loss)
        evaluate_dist()
        
        descr_dict = deepcopy(curr_loss)
        descr_dict['l_de'] = distances[-1]
        descr = ' '.join(['%s:%.2f' % (x, y) for x, y in descr_dict.items()])
        pbar.set_description(descr)
        pbar.update(1)

    # final evaluation
    losses = arr_of_dicts_to_dict_of_arrays(losses)
    loss = model.evaluate(xs_e, ys_e, verbose=0)
    losses['l_fit'].append(loss)


    models = {'model': model, 'decoder': decoder,
              'reconstructor': reconstructor}
    weights = {x: y.get_weights() for x, y in models.items()}

    return losses, distances, weights

def fit_test_model(xs, ys, A):
    """Test the data."""
    # checking that data is correctly generated
    m = tf.keras.Sequential([tf.keras.layers.Dense(2)])
    m.compile('adam', 'mse')
    h = m.fit(xs, ys, epochs=100, verbose=0)
    plt.plot(h.history['loss'])
    print(m.evaluate(xs, ys))
    print(m.weights)
    print(matrix_dist(m.weights[0].numpy().T, A))
