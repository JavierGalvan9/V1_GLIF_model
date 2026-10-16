# Distributed GPU training: investigation summary

This page records the findings that shaped the production multi-GPU path in
[`multi_training.py`](../multi_training.py). The temporary profiling harness,
raw run outputs, and superseded TensorFlow 2.15 prototypes were removed after
the investigation.

## Production guidance

- On a single node, `MirroredStrategy` is the default: one process, one model
  build, and one replica per visible GPU. `multi_worker` remains an opt-in
  alternative when its small measured speed advantage is useful.
- Set a useful **per-replica** batch on every GPU. The recurrent CUDA kernels
  have a fast path specialized for batch size 32; dividing a fixed global
  batch across more GPUs can therefore reduce throughput. For example, at
  66,652 neurons, two GPUs with 32 samples each (global batch 64) reached
  34.0 samples/s versus 19.2 samples/s on one GPU with batch 32, with about
  17.3 GiB peak memory per GPU.
- The convenience option `--global_batch_size` divides the requested batch
  evenly across replicas. Keep the grating/gray split valid for each replica.

## What the investigation found

The first TensorFlow 2.21 multi-GPU runs exposed several separate issues:

1. Connectivity tensors were copied from one GPU to the others repeatedly
   inside recurrent time loops. Device-local CSR resources removed that traffic
   and balanced GPU memory.
2. The CSR resource operators rejected connectivity without a backward pair
   projection, even where backward was not needed. The invariant is now checked
   by the backward operators that require the projection.
3. Keras 3 loss scaling could make different replicas disagree about whether
   gradients were finite. Synchronizing the finiteness decision prevents
   divergent updates and unmatched collectives.
4. One resource-operator build omitted an architecture-specific compile flag,
   selecting a slower backward kernel. Sharing the flag definition across
   builds fixed the mismatch.

A later 203,816-neuron scaling investigation found two additional sources of
multi-GPU overhead: neuron constants were fetched from GPU 0 inside the time
loops, and the chief GPU generated every replica's grating input. Mirrored
per-replica constants and per-GPU grating generation removed those bottlenecks.
At sequence length 500 and batch 32 per replica, measured steps were 2.89 s on
two GPUs (22.2 samples/s) and 2.96 s on four GPUs (43.2 samples/s), compared
with 2.80 s on one GPU. These figures describe the measured RTX PRO 6000
Blackwell setup and are not hardware-independent guarantees.

The main operating lesson is that scaling depends on the model's batch-size
specialization and input workload as well as on gradient communication. NCCL
all-reduce was a small part of the measured step time; cross-device tensor
placement and work concentrated on GPU 0 were the larger initial costs.

## Where to look in the code

- `v1_model_utils/cuda_csr_resources/`: per-device connectivity resources and
  their custom operators.
- `v1_model_utils/optimizers.py`: synchronized gradient-finiteness handling.
- `v1_model_utils/training_orchestration.py`: batch and distributed-launch
  configuration.
- `tests/test_distributed_loss_scaling.py`,
  `tests/test_distributed_csr_resources.py`,
  `tests/test_distribution_strategy.py`, and
  `tests/test_training_orchestration.py`: regression coverage for the
  distributed path.
