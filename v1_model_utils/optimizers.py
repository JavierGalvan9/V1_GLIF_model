
import tensorflow as tf
from math import pi


def _uses_keras_3():
    """Return whether TensorFlow exposes the standalone Keras 3 API."""
    version = getattr(tf.keras, "version", None)
    return version is not None and int(version().split(".", 1)[0]) >= 3

# import tensorflow.compat.v2 as tf

# from keras.optimizers import optimizer
# from keras.saving.object_registration import register_keras_serializable

# # Same TF2.15 / Keras 2.12 public export for consistency
# from tensorflow.python.util.tf_export import keras_export


def build_learning_rate(flags):
    if flags.lr_schedule == "none":
        print(f"Learning-rate schedule: none (constant lr={flags.learning_rate:.6g})")
        return flags.learning_rate

    if flags.lr_schedule == "warmup_cosine":
        schedule = LinearWarmupCosineDecay(
            warmup_start_lr=flags.lr_warmup_start_lr,
            warmup_target_lr=flags.lr_warmup_target_lr,
            warmup_steps=flags.lr_warmup_steps,
            cosine_steps=flags.lr_cosine_steps,
            min_lr=flags.lr_cosine_min_lr,
        )
        print(
            "Learning-rate schedule: warmup_cosine "
            f"(warmup: {flags.lr_warmup_start_lr:.6g}->{flags.lr_warmup_target_lr:.6g} "
            f"in {flags.lr_warmup_steps} steps, cosine: "
            f"{flags.lr_warmup_target_lr:.6g}->{flags.lr_cosine_min_lr:.6g} "
            f"in {flags.lr_cosine_steps} steps)"
        )
        return schedule

    raise ValueError(
        f"Invalid lr_schedule '{flags.lr_schedule}'. "
        "Supported values are: 'none', 'warmup_cosine'."
    )


def create_optimizer(flags, learning_rate, trainable_variables, mixed_precision_module=None):
    keras_3_loss_scaling = flags.dtype == "float16" and _uses_keras_3()
    optimizer_kwargs = {}
    if keras_3_loss_scaling and getattr(flags, "global_clipnorm", 0) > 0:
        # Keras 3 unscales gradients inside LossScaleOptimizer, so clipping
        # belongs on the wrapped optimizer and happens after unscaling.
        optimizer_kwargs["global_clipnorm"] = flags.global_clipnorm
    if flags.optimizer == "adam":
        base_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, epsilon=1e-11, **optimizer_kwargs
        )
    elif flags.optimizer == "exp_adam":
        base_optimizer = ExponentiatedAdam(
            learning_rate=learning_rate, epsilon=1e-11, **optimizer_kwargs
        )
    elif flags.optimizer == "sgd":
        base_optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=0.0, nesterov=False,
            **optimizer_kwargs,
        )
    else:
        print(f"Invalid optimizer: {flags.optimizer}")
        raise ValueError

    if flags.dtype == "float16":
        # Prevent gradient underflow in mixed-float16 training.
        if mixed_precision_module is None:
            from tensorflow.keras import mixed_precision as mixed_precision_module

        base_optimizer = mixed_precision_module.LossScaleOptimizer(base_optimizer)

    # Both the wrapper and its inner optimizer must have created their state
    # before TensorFlow can match optimizer slots from a checkpoint.
    base_optimizer.build(trainable_variables)

    return base_optimizer


def optimizer_supports_loss_scaling(optimizer):
    return hasattr(optimizer, "scale_loss") or (
        hasattr(optimizer, "get_scaled_loss")
        and hasattr(optimizer, "get_unscaled_gradients")
    )


def scale_loss_for_optimizer(optimizer, loss):
    if hasattr(optimizer, "scale_loss"):
        return optimizer.scale_loss(loss)
    if hasattr(optimizer, "get_scaled_loss"):
        return optimizer.get_scaled_loss(loss)
    return loss


def unscale_gradients_for_optimizer(optimizer, gradients):
    # Keras 3 LossScaleOptimizer performs this step inside apply_gradients.
    if hasattr(optimizer, "scale_loss"):
        return gradients
    if hasattr(optimizer, "get_unscaled_gradients"):
        return optimizer.get_unscaled_gradients(gradients)
    return gradients


def synchronize_gradient_finiteness(gradients):
    """Make every replica agree on whether this step's gradients are finite.

    Keras' ``LossScaleOptimizer`` tests finiteness on each replica's own
    gradients and then branches on the result. The finite branch contains the
    gradient all-reduce and the variable update; the non-finite branch only
    halves the loss scale. Nothing reconciles that predicate across replicas,
    so when one replica overflows and another does not they take different
    branches: one replica issues a collective the other never joins, and the
    mirrored loss-scale variable receives two conflicting assignments. Both
    outcomes corrupt the model, which shows up as a loss that turns into NaN
    within the first few updates of a multi-GPU run.

    Overflow is expected under dynamic loss scaling - the scale starts high on
    purpose and is halved until it fits - so this is not a rare corner. It only
    bites replicated training because a single replica never disagrees with
    itself.

    Propagating one replica's overflow to all of them makes the branch decision
    identical everywhere: the update is skipped and the loss scale halved once,
    exactly as on a single GPU. Only one gradient has to carry the signal,
    because the optimizer's check reduces over all of them.
    """
    replica_context = tf.distribute.get_replica_context()
    if replica_context is None or replica_context.num_replicas_in_sync == 1:
        return gradients
    present = [
        index for index, gradient in enumerate(gradients) if gradient is not None
    ]
    if not present:
        return gradients

    def dense_values(gradient):
        """A sparse gradient's finiteness lives in its values."""
        return (
            gradient.values
            if isinstance(gradient, tf.IndexedSlices)
            else gradient
        )

    local_overflow = tf.cast(
        tf.logical_not(
            tf.reduce_all(
                [
                    tf.reduce_all(
                        tf.math.is_finite(
                            tf.cast(dense_values(gradients[index]), tf.float32)
                        )
                    )
                    for index in present
                ]
            )
        ),
        tf.float32,
    )
    replica_overflows = replica_context.all_reduce(
        tf.distribute.ReduceOp.SUM, local_overflow
    )
    signal = tf.where(
        replica_overflows > 0.0,
        tf.constant(float("nan"), tf.float32),
        tf.constant(0.0, tf.float32),
    )

    # Prefer a dense carrier; adding to a sparse gradient's values keeps it
    # sparse but only reaches the rows it already holds, which is still enough
    # because the optimizer reduces over every gradient.
    carrier = next(
        (
            index
            for index in present
            if not isinstance(gradients[index], tf.IndexedSlices)
        ),
        present[0],
    )
    synchronized = list(gradients)
    gradient = gradients[carrier]
    if isinstance(gradient, tf.IndexedSlices):
        synchronized[carrier] = tf.IndexedSlices(
            gradient.values + tf.cast(signal, gradient.values.dtype),
            gradient.indices,
            gradient.dense_shape,
        )
    else:
        synchronized[carrier] = gradient + tf.cast(signal, gradient.dtype)
    return synchronized


def clip_gradients_by_global_norm(gradients, clip_norm, optimizer=None):
    """Clip present gradients together and return their pre-clip global norm.

    A nonpositive limit disables clipping while retaining norm measurement.
    Missing gradients keep their positions so variables remain aligned.
    """
    present_indices = [index for index, gradient in enumerate(gradients) if gradient is not None]
    present = [gradients[index] for index in present_indices]
    global_norm = tf.linalg.global_norm(present)
    if optimizer is not None and hasattr(optimizer, "scale_loss"):
        # Keras 3's wrapper will unscale and delegate clipping to the inner
        # optimizer configured by create_optimizer().
        dynamic_scale = getattr(optimizer, "dynamic_scale", None)
        if dynamic_scale is not None:
            global_norm /= tf.cast(dynamic_scale, global_norm.dtype)
        return gradients, global_norm
    if clip_norm <= 0:
        return gradients, global_norm

    clipped_present, _ = tf.clip_by_global_norm(present, clip_norm, use_norm=global_norm)
    clipped = list(gradients)
    for index, gradient in zip(present_indices, clipped_present):
        clipped[index] = gradient
    return clipped, global_norm


@tf.keras.utils.register_keras_serializable(package="V1GLIF")
class LinearWarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay."""

    def __init__(
        self,
        warmup_start_lr=0.1,
        warmup_target_lr=0.05,
        warmup_steps=100,
        cosine_steps=900,
        min_lr=0.001,
        name="LinearWarmupCosineDecay",
    ):
        super().__init__()
        self.warmup_start_lr = float(warmup_start_lr)
        self.warmup_target_lr = float(warmup_target_lr)
        self.warmup_steps = int(warmup_steps)
        self.cosine_steps = int(cosine_steps)
        self.min_lr = float(min_lr)
        self.name = name

        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}.")
        if self.cosine_steps <= 0:
            raise ValueError(f"cosine_steps must be > 0, got {self.cosine_steps}.")
        if self.min_lr < 0.0:
            raise ValueError(f"min_lr must be >= 0, got {self.min_lr}.")

        # Pre-create scalar constants once to avoid rebuilding/casting every call.
        self._warmup_start_lr_tf = tf.constant(self.warmup_start_lr, dtype=tf.float32)
        self._warmup_target_lr_tf = tf.constant(self.warmup_target_lr, dtype=tf.float32)
        self._min_lr_tf = tf.constant(self.min_lr, dtype=tf.float32)
        self._warmup_steps_tf = tf.constant(float(self.warmup_steps), dtype=tf.float32)
        self._warmup_den_tf = tf.constant(float(max(1, self.warmup_steps - 1)), dtype=tf.float32)
        self._cosine_steps_tf = tf.constant(float(self.cosine_steps), dtype=tf.float32)
        self._pi_tf = tf.constant(pi, dtype=tf.float32)

    def __call__(self, step):
        with tf.name_scope(self.name):
            step = tf.cast(step, tf.float32)

            def warmup_branch():
                if self.warmup_steps > 1:
                    warmup_progress = tf.clip_by_value(step / self._warmup_den_tf, 0.0, 1.0)
                    return self._warmup_start_lr_tf + (
                        self._warmup_target_lr_tf - self._warmup_start_lr_tf
                    ) * warmup_progress
                return self._warmup_target_lr_tf

            def cosine_branch():
                cosine_step = tf.maximum(step - self._warmup_steps_tf, 0.0)
                cosine_progress = tf.clip_by_value(cosine_step / self._cosine_steps_tf, 0.0, 1.0)
                cosine_decay = 0.5 * (1.0 + tf.cos(self._pi_tf * cosine_progress))
                return self._min_lr_tf + (self._warmup_target_lr_tf - self._min_lr_tf) * cosine_decay

            if self.warmup_steps > 0:
                return tf.cond(step < self._warmup_steps_tf, warmup_branch, cosine_branch)
            return cosine_branch()

    def get_config(self):
        return {
            "warmup_start_lr": self.warmup_start_lr,
            "warmup_target_lr": self.warmup_target_lr,
            "warmup_steps": self.warmup_steps,
            "cosine_steps": self.cosine_steps,
            "min_lr": self.min_lr,
            "name": self.name,
        }


class ExponentiatedAdam(tf.keras.optimizers.Optimizer):
    r"""Adam-like optimizer with *exponentiated* gradient updates.

    This rewrites the final update from

        w <- w - alpha * m / (sqrt(v) + epsilon)

    to

        w <- w * exp(- alpha * m / (sqrt(v) + epsilon) * sign(w)),

    preserving the rest of the Adam algorithm (moments `m`, `v`, AMSGrad, etc.).
    By default, `sign(0) = +1` so zero-valued parameters can still move off zero.

    Sparse updates are applied consistently via `scatter_mul()` on the affected
    slices only.

    Reference:
      - "Brain-like learning with exponentiated gradients" (and related papers).
      - The original Adam reference:
        [Kingma et al., 2014](http://arxiv.org/abs/1412.6980).
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="ExponentiatedAdam",
        **kwargs
    ):
        """Create a new ExponentiatedAdam optimizer."""
        optimizer_kwargs = dict(
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            **kwargs
        )
        if _uses_keras_3():
            super().__init__(learning_rate=learning_rate, **optimizer_kwargs)
        else:
            super().__init__(jit_compile=jit_compile, **optimizer_kwargs)
            self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

    def build(self, var_list):
        """Initialize optimizer variables.

        Similar to Adam: we have slot variables for
        - m (first moment)
        - v (second moment)
        - vhat (if amsgrad=True, for the max of second moments).
        """
        if getattr(self, "built", False) or getattr(self, "_built", False):
            return
        super().build(var_list)
        self._momentums = []
        self._velocities = []
        for var in var_list:
            if _uses_keras_3():
                self._momentums.append(
                    self.add_variable_from_reference(var, name="m")
                )
                self._velocities.append(
                    self.add_variable_from_reference(var, name="v")
                )
            else:
                self._momentums.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="m"
                    )
                )
                self._velocities.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="v"
                    )
                )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                if _uses_keras_3():
                    slot = self.add_variable_from_reference(var, name="vhat")
                else:
                    slot = self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    )
                self._velocity_hats.append(slot)
        if not _uses_keras_3():
            self._built = True

    def _coalesce_indexed_slices(self, grad, variable_dtype=None, index_dtype=tf.int32):
        """Coalesce duplicate indices in an IndexedSlices gradient."""
        indices = tf.cast(grad.indices, index_dtype)
        values = grad.values
        if variable_dtype is not None and values.dtype != variable_dtype:
            values = tf.cast(values, variable_dtype)

        unique_idx, inverse_pos = tf.unique(indices, out_idx=tf.int32)
        num_unique = tf.shape(unique_idx, out_type=tf.int32)[0]
        coalesced_values = tf.math.unsorted_segment_sum(values, inverse_pos, num_unique)

        dense_shape = grad.dense_shape
        if dense_shape is None:
            dense_shape = tf.cast(tf.shape(values)[0:1], index_dtype)
        else:
            dense_shape = tf.cast(dense_shape, index_dtype)

        return tf.IndexedSlices(coalesced_values, unique_idx, dense_shape)

    def _variable_index(self, variable):
        """Slot index for `variable`, which may arrive as a raw `tf.Variable`.

        Under `tf.distribute`, Keras 3 hands `update_step` the per-replica
        `tf.Variable` rather than the Keras `Variable`, so the lookup has to go
        through the optimizer's own variable keying instead of a name/path map.
        """
        if _uses_keras_3():
            return self._get_variable_index(variable)
        return self._index_dict[self._var_key(variable)]

    def _assign(self, variable, value):
        if _uses_keras_3():
            self.assign(variable, value)
        else:
            variable.assign(value)

    def _assign_add(self, variable, value):
        if _uses_keras_3():
            self.assign_add(variable, value)
        else:
            variable.assign_add(value)

    # No `tf.function` here: Keras 3 calls `update_step` with a Keras `Variable`,
    # which is not a traceable type, and the caller is already inside a graph.
    def update_step(self, gradient, variable, learning_rate=None):
        """Update step given gradient and the associated model variable."""
        # if isinstance(gradient, tf.IndexedSlices):
        #     gradient = tf.convert_to_tensor(gradient)
        # Get current iteration
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        # Compute powers of beta_1 and beta_2
        beta_1_t = tf.cast(self.beta_1, variable.dtype)
        beta_2_t = tf.cast(self.beta_2, variable.dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)

        # Get the current learning rate (supports schedules)
        if learning_rate is None:
            learning_rate = self.learning_rate
        lr = tf.cast(learning_rate, variable.dtype)
        # Standard Adam alpha correction
        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        # Fetch the slot variables for this parameter
        variable_index = self._variable_index(variable)
        m = self._momentums[variable_index]
        v = self._velocities[variable_index]

        # ------------------------
        # Sparse vs. Dense branch
        # ------------------------
        if isinstance(gradient, tf.IndexedSlices):
            gradient = self._coalesce_indexed_slices(
                gradient,
                variable_dtype=variable.dtype,
                index_dtype=tf.int32,
            )

            # Sparse gradient
            # 1) Update m (first moment)
            scatter_indices = tf.expand_dims(gradient.indices, axis=1)
            m_update = tf.tensor_scatter_nd_add(
                tf.zeros_like(m),
                scatter_indices,
                gradient.values * (1 - beta_1_t),
            )
            self._assign(m, beta_1_t * m + m_update)
            # 2) Update v (second moment)
            v_update = tf.tensor_scatter_nd_add(
                tf.zeros_like(v),
                scatter_indices,
                tf.square(gradient.values) * (1 - beta_2_t),
            )
            self._assign(v, beta_2_t * v + v_update)
            # 3) AMSGrad if needed
            if self.amsgrad:
                vhat = self._velocity_hats[variable_index]
                self._assign(vhat, tf.maximum(vhat, v))
                v_used = vhat
            else:
                v_used = v

            # 4) Exponentiated update for only the slices that changed
            # Gather the relevant slices for var, m, v
            var_slices = tf.gather(variable, gradient.indices)
            m_slices = tf.gather(m, gradient.indices)
            v_slices = tf.gather(v_used, gradient.indices)

            # adam_grad = m / (sqrt(v)+eps)
            adam_grad_slices = m_slices / (tf.sqrt(v_slices) + self.epsilon)

            # sign(w) for these slices, fallback sign(0)=+1
            sign_w_slices = tf.sign(var_slices)
            sign_w_slices = tf.where(
                tf.equal(sign_w_slices, 0), tf.ones_like(sign_w_slices), sign_w_slices
            )

            # exponent = - alpha * adam_grad_slices * sign_w_slices
            exponent_slices = -alpha * adam_grad_slices * sign_w_slices

            # multiplier = exp(exponent_slices)
            multiplier_slices = tf.exp(exponent_slices)

            # var_slices_new = var_slices * multiplier_slices
            # We can do partial update with scatter_mul:
            #   new_var[i] = old_var[i] * multiplier[i]
            multipliers = tf.tensor_scatter_nd_update(
                tf.ones_like(variable), scatter_indices, multiplier_slices
            )
            self._assign(variable, variable * multipliers)

        else:
            # Dense gradient
            # 1) Update m
            self._assign_add(m, (gradient - m) * (1 - beta_1_t))
            # 2) Update v
            self._assign_add(v, (tf.square(gradient) - v) * (1 - beta_2_t))
            # 3) AMSGrad
            if self.amsgrad:
                vhat = self._velocity_hats[variable_index]
                self._assign(vhat, tf.maximum(vhat, v))
                v_used = vhat
            else:
                v_used = v

            # 4) Exponentiated update
            adam_grad = m / (tf.sqrt(v_used) + self.epsilon)

            # sign(w), fallback sign(0)=+1
            sign_w = tf.sign(variable)
            sign_w = tf.where(tf.equal(sign_w, 0), tf.ones_like(sign_w), sign_w)

            exponent = -alpha * adam_grad * sign_w
            multiplier = tf.exp(exponent)
            self._assign(variable, variable * multiplier)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config
