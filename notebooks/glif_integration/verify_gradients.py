"""Finite-difference check of the GLIF state gradients, on both backends.

BPTT differentiates the step, so the propagator is only as good as its
Jacobian. The TensorFlow path is plain ops and is differentiated by autodiff;
the CUDA path carries a hand-written backward kernel. This script checks both
against central finite differences of the forward step.

Two inputs are deliberately excluded, because a finite difference there
measures a step function rather than the transition:

* the emitted spikes, which come from a Heaviside with a surrogate gradient;
* the delayed spike buffer, which feeds the event-driven recurrent kernel,
  where a spike is consumed as a binary event.

Both are pre-existing choices of the model and independent of the integrator.
What is left - v, the two synaptic state blocks, the ASCs and prev_z - is
smooth, and is exactly what the propagator computes.

Writes ``gradient_check.json``; needs TensorFlow and a GPU for the cuda backend.
"""
import json, os, pickle, sys, tempfile
import numpy as np, tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
# V1Column reads synaptic_data/tau_basis.npy by a path relative to the process
# working directory, so the check has to run from the repository root.
os.chdir(REPO)
from v1_model_utils import cuda_csr_recurrent, spatial_layout
from v1_model_utils.models import V1Column

N = 8


def cell(tmp, backend, scheme, detach_reset=True, detach_asc_reset=False,
         dampening=0.5):
    td = os.path.join(tmp, "tf_data"); os.makedirs(td, exist_ok=True)
    pickle.dump({0: np.ones(4, np.float32)},
                open(os.path.join(td, "syn_id_to_syn_weights_dict.pkl"), "wb"))
    params = dict(V_th=np.ones(1, np.float32), E_L=np.zeros(1, np.float32),
                  V_reset=np.zeros(1, np.float32), g=np.full(1, 8.0, np.float32),
                  C_m=np.full(1, 115.0, np.float32), t_ref=np.full(1, 3, np.float32),
                  k=np.array([[0.003, 0.3]], np.float32),
                  asc_amps=np.array([[-110.0, 243.0]], np.float32))
    net = dict(data_dir=tmp, n_nodes=N, node_type_ids=np.zeros(N, np.int32),
               node_params=params,
               # Zero recurrent weights: the spike buffer then reaches the loss
               # only through the reset and the ASC, which is the path under
               # test. A live recurrent synapse would route it through the
               # event-driven CSR kernel, where a spike is a binary event and a
               # finite difference measures a step function instead.
               synapses=dict(indices=np.array([[0, 0], [1, 1]], np.int32),
                             weights=np.zeros(2, np.float32),
                             delays=np.array([3, 1], np.float32),
                             syn_ids=np.zeros(2, np.uint8), dense_shape=(N, N)))
    ext = dict(n_inputs=2, indices=np.array([[0, 0], [1, 1]], np.int32),
               weights=np.array([.1, .2], np.float32), delays=np.ones(2, np.float32),
               syn_ids=np.zeros(2, np.uint8))
    # The CUDA kernels index weights by CSR position, so reorder the fixture's
    # edges the same way the training path reorders a loaded network.
    orders = None
    if cuda_csr_recurrent.DIRECT_CSR:
        net, ext, bkg, orders = spatial_layout.apply_csr_edge_order(net, ext, ext)
    else:
        bkg = ext
    return V1Column(net, ext, bkg, edge_orders=orders,
                    batch_size=2, acceleration=backend,
                    train_recurrent=False, train_input=False, train_noise=False,
                    integration_scheme=scheme, detach_reset=detach_reset,
                    detach_asc_reset=detach_asc_reset,
                    recurrent_dampening_factor=dampening)


def state_of(c, rng, batch=2):
    """A state with everything non-trivial: spikes, voltage, currents."""
    s = list(c.zero_state(batch, tf.float32))
    s[0] = tf.constant((rng.random(s[0].shape) < 0.3).astype(np.float32))
    s[1] = tf.constant(rng.uniform(-0.2, 0.9, s[1].shape).astype(np.float32))
    s[2] = tf.zeros_like(s[2])
    for i in (3, 4, 5):
        s[i] = tf.constant(rng.uniform(-0.3, 0.3, s[i].shape).astype(np.float32))
    return s


def loss_of(c, inputs, state, weights):
    """A scalar over the *continuous* state only.

    The new spike history (state element 0) is deliberately excluded: spikes come
    from a Heaviside with a surrogate gradient, so finite differences across it
    measure the surrogate's deviation from the true derivative rather than any
    error in the state transition. What is left - v, asc, psc_rise, psc - is a
    smooth function of the inputs, and is exactly what the propagator computes.
    """
    _, new_state = c(inputs, tuple(state))
    parts = [new_state[i] for i in (1, 3, 4, 5)]
    return sum(w * tf.reduce_sum(tf.cast(x, tf.float32) * m)
               for w, x, m in zip(weights, parts, MASKS))


MASKS = []


def check(backend, scheme, detach_reset=False, detach_asc_reset=False,
          dampening=1.0, eps=3e-3, probes=24, seed=11):
    """Return the worst relative FD discrepancy per differentiable state block."""
    global MASKS
    tf.keras.mixed_precision.set_global_policy("float32")
    rng = np.random.default_rng(seed)
    c = cell(tempfile.mkdtemp(), backend, scheme, detach_reset=detach_reset,
             detach_asc_reset=detach_asc_reset, dampening=dampening)
    N_ = c._n_neurons
    state = state_of(c, rng)
    inputs = tf.constant(rng.uniform(0, 1, (2, c.input_dim)).astype(np.float32))

    _, probe = c(inputs, tuple(state))
    floats = [probe[i] for i in (1, 3, 4, 5)]
    MASKS = [tf.constant(rng.uniform(0.3, 1.7, x.shape).astype(np.float32))
             for x in floats]
    weights = [1.0 + 0.4 * i for i in range(len(floats))]

    names = {0: "prev_z", 1: "v", 3: "asc", 4: "psc_rise", 5: "psc"}
    watched = {i: tf.Variable(state[i]) for i in names}
    with tf.GradientTape() as tape:
        s = list(state)
        for i, v in watched.items():
            s[i] = v
        loss = loss_of(c, inputs, s, weights)
    analytic = tape.gradient(loss, list(watched.values()))

    out = {}
    for (idx, name), grad in zip(names.items(), analytic):
        g = np.asarray(grad)
        base = np.asarray(state[idx], np.float32)
        flat = base.reshape(-1)
        if idx == 0:
            pool = np.array([b * base.shape[1] + col
                             for b in range(base.shape[0]) for col in range(N_)])
        else:
            pool = np.arange(flat.size)
        picks = rng.choice(pool, size=min(probes, pool.size), replace=False)
        fd = np.zeros_like(flat)
        for k in picks:
            for sign in (+1, -1):
                bumped = flat.copy(); bumped[k] += sign * eps
                s = list(state); s[idx] = tf.constant(bumped.reshape(base.shape))
                fd[k] += sign * float(loss_of(c, inputs, s, weights))
        fd /= 2 * eps
        gf = g.reshape(-1)
        err = np.abs(fd[picks] - gf[picks]).max()
        scale = max(np.abs(gf[picks]).max(), 1e-8)
        out[name] = dict(analytic=float(np.abs(gf[picks]).max()),
                         abs_err=float(err), rel_err=float(err / scale))
    return out


def main():
    results = {}
    backends = ["tensorflow"]
    if tf.config.list_physical_devices("GPU"):
        backends.append("cuda")
    for backend in backends:
        for scheme in ("euler", "exact"):
            key = f"{backend}/{scheme}"
            results[key] = check(backend, scheme)
            worst = max(v["rel_err"] for v in results[key].values())
            print(f"{key:<20} worst relative FD discrepancy {worst:.2e}")
            for name, v in results[key].items():
                print(f"    {name:<10} |analytic| {v['analytic']:>10.4f}   "
                      f"rel {v['rel_err']:.2e}")
    path = os.path.join(HERE, "gradient_check.json")
    with open(path, "w") as handle:
        json.dump(results, handle, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
