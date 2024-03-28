import os
import sys
import json
import numpy as np
import tensorflow as tf
import datetime as dt
from time import time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib import pyplot as plt
import stim_dataset
from v1_model_utils.plotting_utils import InputActivityFigure, ModelMetricsAnalysis


def printgpu(verbose=0):
    if tf.config.list_physical_devices('GPU'):
        meminfo = tf.config.experimental.get_memory_info('GPU:0')
        current = meminfo['current'] / 1024**3
        peak = meminfo['peak'] / 1024**3
        if verbose == 0:
            print(f"GPU memory use: {current:.2f} GB / Peak: {peak:.2f} GB")
        if verbose == 1:
            return current, peak

def compose_str(metrics_values):
        _acc, _loss, _rate, _rate_loss, _voltage_loss, _osi_loss = metrics_values
        _s = f'Loss {_loss:.4f}, '
        _s += f'RLoss {_rate_loss:.4f}, '
        _s += f'VLoss {_voltage_loss:.4f}, '
        _s += f'OLoss {_osi_loss:.4f}, '
        _s += f'Accuracy {_acc:.4f}, '
        _s += f'Rate {_rate:.4f}'
        return _s


class Callbacks:
    def __init__(self, model, optimizer, distributed_roll_out, network, flags, logdir, strategy, 
                metrics_keys, pre_delay=50, post_delay=50):
        self.network = network
        self.flags = flags
        self.logdir = logdir
        # self.manager = manager
        self.osi_dsi_data_set = strategy.distribute_datasets_from_function(self.get_osi_dsi_dataset_fn(regular=True))
        self.distributed_roll_out = distributed_roll_out
        self.metrics_keys = metrics_keys
        self.pre_delay = pre_delay
        self.post_delay = post_delay
        self.epoch = 0
        self.step = 0
        self.no_improve_epochs = 0
        # please create a dictionary to save the values of the metric keys after each epoch
        self.epoch_metric_values = {key: [] for key in self.metrics_keys}
        # self.epoch_metric_values['val_osi_dsi_loss'] = []
        # self.step_counter = tf.Variable(0, dtype=tf.int64)
        self.step_running_time = []
        self.min_val_loss = float('inf')
        self.initial_metric_values = None
        self.summary_writer = tf.summary.create_file_writer(self.logdir)
        with open(os.path.join(self.logdir, 'config.json'), 'w') as f:
            json.dump(flags.flag_values_dict(), f, indent=4)
    
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        if flags.restore_from != '':
            with strategy.scope():
                # checkpoint.restore(tf.train.latest_checkpoint(flags.restore_from))
                checkpoint.restore(self.flags.restore_from)
                print(f'Model parameters of {self.flags.task_name} restored from {self.flags.restore_from}')

        self.manager = tf.train.CheckpointManager(
                                            checkpoint, directory=self.logdir, max_to_keep=1, # just keep the best model
                                            # checkpoint_interval = 5, # save ckpt for data analysis
                                            # step_counter=self.step_counter
                                        )

    def get_osi_dsi_dataset_fn(self, regular=False):
        def _f(input_context):
            post_delay = self.flags.seq_len - (2500 % self.flags.seq_len)
            _data_set = stim_dataset.generate_drifting_grating_tuning(
                seq_len=2500+post_delay,
                pre_delay=500,
                post_delay = post_delay,
                n_input=self.flags.n_input,
                regular=regular
            ).batch(1)
                        
            return _data_set
        return _f

    def on_train_begin(self):
        print("---------- Training started at ", dt.datetime.now().strftime('%d-%m-%Y %H:%M'), ' ----------\n')
        self.train_start_time = time()

    def on_train_end(self, metric_values):
        self.train_end_time = time()
        self.final_metric_values = metric_values
        print("\n ---------- Training ended at ", dt.datetime.now().strftime('%d-%m-%Y %H:%M'), ' ----------\n')
        print(f"Total time spent: {self.train_end_time - self.train_start_time:.2f} seconds")
        print(f"Average step time: {np.mean(self.step_running_time):.2f} seconds\n")
        # Determine the maximum key length for formatting the table
        max_key_length = max(len(key) for key in self.metrics_keys)

        # Start of the Markdown table
        print(f"| {'Metric':<{max_key_length}} | {'Initial Value':<{max_key_length}} | {'Final Value':<{max_key_length}} |")
        print(f"|{'-' * (max_key_length + 2)}|{'-' * (max_key_length + 2)}|{'-' * (max_key_length + 2)}|")


        n_metrics = len(self.initial_metric_values)//2
        for initial, final, key in zip(self.initial_metric_values[n_metrics:], self.final_metric_values[n_metrics:], self.metrics_keys[n_metrics:]):
            print(f"| {key:<{max_key_length}} | {initial:<{max_key_length}.3f} | {final:<{max_key_length}.3f} |")

    def on_epoch_start(self):
        self.epoch += 1
        # self.step_counter.assign_add(1)
        self.epoch_init_time = time()
        date_str = dt.datetime.now().strftime('%d-%m-%Y %H:%M')
        print(f'Epoch {self.epoch:2d}/{self.flags.n_epochs} @ {date_str}')
        tf.print(f'\nEpoch {self.epoch:2d}/{self.flags.n_epochs} @ {date_str}')

    def on_epoch_end(self, z, x, y, metric_values, verbose=True):
        self.step = 0
        if self.initial_metric_values is None:
            self.initial_metric_values = metric_values
        
        if verbose:
            print_str = '  Validation: \n' 
            val_values = metric_values[len(metric_values)//2:]
            print_str += '    ' + compose_str(val_values) 
            print(print_str)
            mem_data = printgpu(verbose=1)
            print(f'    Memory consumption (current - peak): {mem_data[0]:.2f} GB - {mem_data[1]:.2f} GB'+ '\n')

        # val_classification_loss = metric_values[6] - metric_values[8] - metric_values[9] 
        # metric_values.append(val_classification_loss)
        self.epoch_metric_values = {key: value + [metric_values[i]] for i, (key, value) in enumerate(self.epoch_metric_values.items())}

        if 'val_loss' in self.metrics_keys:
            val_loss_index = self.metrics_keys.index('val_loss')
            val_loss_value = metric_values[val_loss_index]
        else:
            val_loss_index = self.metrics_keys.index('train_loss')
            val_loss_value = metric_values[val_loss_index]

        self.plot_losses_curves()

        # if val_loss_value < self.min_val_loss:
        if True:
            self.min_val_loss = val_loss_value
            self.no_improve_epochs = 0

            # self.plot_lgn_activity(x)

            t0 = time()
            self.plot_raster(z, x, y)
            print('Raster plot time:', time()-t0)

            t0 = time()
            self.plot_mean_firing_rate_boxplot(z, y)
            print('Mean firing rate boxplot time:', time()-t0)

            self.save_model()
        else:
            self.no_improve_epochs += 1
           
        if (self.epoch) % 50 == 0:
            t0 = time()
            self.plot_osi_dsi()
            print('OSI and DSI plot time:', (time()-t0))

        with self.summary_writer.as_default():
            for k, v in zip(self.metrics_keys, metric_values):
                tf.summary.scalar(k, v, step=self.epoch)

        # EARLY STOPPING CONDITIONS
        if (0 < self.flags.max_time < (time() - self.epoch_init_time) / 3600):
            print(f'[ Maximum optimization time of {self.flags.max_time:.2f}h reached ]')
            stop = True
        elif self.no_improve_epochs >= 500:
            print("Early stopping: Validation loss has not improved for 50 epochs.")
            stop = True  
        else:
            stop = False

        return stop


    def on_step_start(self):
        self.step += 1
        self.step_init_time = time()

    def on_step_end(self, train_values, y, verbose=True):
        self.step_running_time.append(time() - self.step_init_time)
        if verbose:
            print_str = f'  Step {self.step:2d}/{self.flags.steps_per_epoch} - Angle: {y[0][0]:.2f}\n'
            print_str += '    ' + compose_str(train_values)
            print(print_str)
            print(f'    Step running time: {time() - self.step_init_time:.2f}s')
            mem_data = printgpu(verbose=1)
            print(f'    Memory consumption (current - peak): {mem_data[0]:.2f} GB - {mem_data[1]:.2f} GB')
        

    def save_model(self):
        # self.step_counter.assign_add(1)
        print(f'[ Saving the model at epoch {self.epoch} ]')
        try:
            p = self.manager.save(checkpoint_number=self.epoch)
            print(f'Model saved in {p}\n')
        except:
            print("Saving failed. Maybe next time?")

    def plot_losses_curves(self):
        # plotting_metrics = ['val_loss', 'val_firing_rate', 'val_rate_loss', 'val_voltage_loss']
        plotting_metrics = ['val_loss', 'val_osi_loss', 'val_rate_loss', 'val_voltage_loss']
        images_dir = os.path.join(self.logdir, 'Loss_curves')
        os.makedirs(images_dir, exist_ok=True)

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        for i, metric_key in enumerate(plotting_metrics):
            ax = axs[i // 2, i % 2]
            ax.plot(range(1, self.epoch + 1), self.epoch_metric_values[metric_key])
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_key)
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, f'losses_curves_epoch.png'), dpi=300, transparent=False)
        plt.close()
    
    def plot_raster(self, z, x, y):
        z = z.numpy()
        x = x.numpy()
        y = y.numpy()
        images_dir = os.path.join(self.logdir, 'Raster_plots')
        os.makedirs(images_dir, exist_ok=True)
        graph = InputActivityFigure(
                                    self.network,
                                    self.flags.data_dir,
                                    images_dir,
                                    filename=f'Epoch_{self.epoch}',
                                    frequency=self.flags.temporal_f,
                                    stimuli_init_time=self.pre_delay,
                                    stimuli_end_time=self.flags.seq_len-self.post_delay,
                                    reverse=False,
                                    plot_core_only=True,
                                    )
        graph(x, z)

    # def plot_lgn_activity(self, x):
    #     x = x.numpy()[0, :, :]
    #     x_mean = np.mean(x, axis=1)
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(x_mean)
    #     plt.title('Mean input activity')
    #     plt.xlabel('Time (ms)')
    #     plt.ylabel('Mean input activity')
    #     plt.savefig(os.path.join(self.logdir, 'Mean_input_activity.png'))


    def plot_mean_firing_rate_boxplot(self, z, y):
        z = z.numpy()
        y = y.numpy()
        boxplots_dir = os.path.join(self.logdir, 'Boxplots')
        os.makedirs(boxplots_dir, exist_ok=True)
        metrics_analysis = ModelMetricsAnalysis(self.network, self.flags.neurons, data_dir=self.flags.data_dir, n_trials=1,
                                                analyze_core_only=True, drifting_gratings_init=self.pre_delay, drifting_gratings_end=self.flags.seq_len-self.post_delay, 
                                                directory=boxplots_dir, filename=f'Epoch_{self.epoch}')
        metrics_analysis(z, y)    

    def plot_osi_dsi(self):
        print('Starting to plot OSI and DSI...')
        sim_duration = (2500//self.flags.seq_len + 1) * self.flags.seq_len
        n_trials_per_angle = 10
        spikes = np.zeros((8, sim_duration, self.flags.neurons), dtype=float)
        DG_angles = np.arange(0, 360, 45)
        for trial_id in range(n_trials_per_angle):
            t0 = time()
            test_it = iter(self.osi_dsi_data_set)
            for angle_id, angle in enumerate(range(0, 360, 45)):
                t0 = time()
                x, y, _, w = next(test_it)
                chunk_size = self.flags.seq_len
                num_chunks = (2500//self.flags.seq_len + 1)
                for i in range(num_chunks):
                    t0 = time()
                    chunk = x[:, i * chunk_size : (i + 1) * chunk_size, :]
                    z_chunk = self.distributed_roll_out(chunk, y, w, output_spikes=True)
                    spikes[angle_id, i * chunk_size : (i + 1) * chunk_size, :] += z_chunk.numpy()[0, :, :].astype(float)

                # spikes[angle_id, :, :] += z.numpy()[0, :, :]
        spikes = spikes/n_trials_per_angle
        # Slice only the first 2500 ms
        spikes = spikes[:, :2500, :]

        images_dir = os.path.join(self.logdir, 'Raster_plots_OSI_DSI')
        os.makedirs(images_dir, exist_ok=True)
        graph = InputActivityFigure(
                                    self.network,
                                    self.flags.data_dir,
                                    images_dir,
                                    filename=f'Epoch_{self.epoch}',
                                    frequency=self.flags.temporal_f,
                                    stimuli_init_time=500,
                                    stimuli_end_time=2500,
                                    reverse=False,
                                    plot_core_only=True,
                                    )
        graph(x[:, :2500, :].numpy(), spikes)
        
        boxplots_dir = os.path.join(self.logdir, 'Boxplots_OSI_DSI')
        os.makedirs(boxplots_dir, exist_ok=True)
        metrics_analysis = ModelMetricsAnalysis(self.network, self.flags.neurons, data_dir=self.flags.data_dir,
                                                drifting_gratings_init=500, drifting_gratings_end=2500,
                                                analyze_core_only=True, directory=boxplots_dir, filename=f'Epoch_{self.epoch}')
        metrics_analysis(spikes, DG_angles)