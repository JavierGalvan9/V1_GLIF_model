# minimal script that utilize memory of the network.
# %%

import tensorflow as tf
from time import time

from v1_model_utils import load_sparse, models, other_v1_utils, toolkit
from v1_model_utils.plotting_utils import InputActivityFigure, LaminarPlot, LGN_sample_plot, PopulationActivity, RasterPlot
import stim_dataset


class Fake():
    def __init__(self):        
        self.neurons = 5000
        self.batch_size = 1
        self.data_dir = 'GLIF_network'
        self.core_only = True
        self.seed = 3000
        self.connected_selection = True
        self.n_output = 2
        self.neurons_per_output = 10
        self.n_input = 17400
        self.seq_len = 600 
        self.delays = '10,10'
        self.voltage_cost = 1.0
        

        
flags = Fake()

# network, lgn_input, bkg_input = load_sparse.load_v1(flags, flags.neurons)
network, lgn_input, bkg_input = load_sparse.cached_load_v1(flags, flags.neurons)

model = models.create_model(
    network,
    lgn_input,
    bkg_input,
    seq_len=flags.seq_len,
    n_input=flags.n_input,
    n_output=flags.n_output,
    cue_duration=50,
    dtype=tf.float32,
    input_weight_scale=1.0,
    dampening_factor=0.5,
    gauss_std=0.28,
    lr_scale=0.01,
    train_input=False,
    train_recurrent=True,
    neuron_output=False,
    recurrent_dampening_factor=0.5,
    batch_size=flags.batch_size,
    pseudo_gauss=False,
    use_state_input=True,
    return_state=True,
)

del lgn_input, bkg_input

model.build((flags.batch_size, flags.seq_len, flags.n_input))

rsnn_layer = model.get_layer("rsnn")
prediction_layer = model.get_layer('prediction')
extractor_model = tf.keras.Model(inputs=model.inputs,
                                 outputs=[rsnn_layer.output, model.output, prediction_layer.output])

# %%
dtype = tf.float32


zero_state = rsnn_layer.cell.zero_state(flags.batch_size)

state_variables = tf.nest.map_structure(lambda a: tf.Variable(
    a, trainable=False, synchronization=tf.VariableSynchronization.ON_READ
), zero_state)

# @tf.function
def roll_out(_x, _y, _w):
    _initial_state = tf.nest.map_structure(lambda _a: _a.read_value(), state_variables)
    dummy_zeros = tf.zeros((flags.batch_size, flags.seq_len, flags.neurons), dtype)

    # v1 = rsnn_layer.cell
    # v1 = extractor_model.get_layer('rsnn').cell
    # del v1.sparse_w_rec
    # v1.prepare_sparse_weight(v1.recurrent_weight_values, v1.recurrent_weights_factors)
    # let's be explicit. calculate the sparse tensor here.
    # sparse_w_recs = []
    # for r_id in range(v1._n_syn_basis):
    #     w_syn_recep = v1.recurrent_weight_values * v1.recurrent_weights_factors[r_id]
    #     sparse_w_rec = tf.sparse.SparseTensor(
    #         v1.recurrent_indices,
    #         tf.cast(w_syn_recep, v1._compute_dtype),
    #         v1.recurrent_dense_shape
    #     )
    #     sparse_w_recs.append(sparse_w_rec)
    
    # # v1.prepare_sparse_weight()

    _out, _p, _ = extractor_model((_x, dummy_zeros, _initial_state))
    # print('roll out time: ', time.time() - stt)

    _z, _v, _input_current = _out[0]
    voltage_32 = (tf.cast(_v, tf.float32) - rsnn_layer.cell.voltage_offset) / rsnn_layer.cell.voltage_scale
    v_pos = tf.square(tf.nn.relu(voltage_32 - 1.))
    v_neg = tf.square(tf.nn.relu(-voltage_32 + 1.))
    voltage_loss = tf.reduce_mean(tf.reduce_sum(v_pos + v_neg, -1)) * flags.voltage_cost
    # rate_loss = rate_distribution_regularizer(_z)
    # classification loss is turned off for now.
    # classification_loss = compute_loss_gratings(_y, _z)

    # _aux = dict(rate_loss=rate_loss, voltage_loss=voltage_loss)
    _aux = dict(rate_loss=0, voltage_loss=voltage_loss)
    # _loss = classification_loss + rate_loss + voltage_loss
    _loss = voltage_loss
    # _loss = rate_loss

    return _out, _p, _loss, _aux

optimizer = tf.keras.optimizers.Adam(0.01, epsilon=1e-11)    

# @tf.function
def train_step(_x, _y, _w):
    # _out, _p, _loss, _aux = roll_out(_x, _y, _w)

    with tf.GradientTape() as tape:
        _out, _p, _loss, _aux = roll_out(_x, _y, _w)
    # tf.print("calculating gradient...")
    _grads = tape.gradient(_loss, model.trainable_variables)

    for g, v in zip(_grads, model.trainable_variables):
        _op = optimizer.apply_gradients([(g, v)])

    return _out, _p, _loss, _aux




# %%
data = stim_dataset.generate_drifting_grating_tuning(
    seq_len=flags.seq_len,
    pre_delay=10,
    post_delay=10,
    n_input=flags.n_input
)

t0 = time()
for value in data.take(1):
    x, y, _, w = value
    break
x = tf.expand_dims(x, 0)
print(f'LGN spikes calculation time: {time() - t0:.2f}s')

# %% run the model
step_t0 = time()
out = train_step(x, y, w)
print(f'Step running time: {time() - step_t0:.2f}s')

print(out[-1]) # gradient
print(out[-2]) # aux loss

# out = roll_out(x, y, w)
# %%

print('done')