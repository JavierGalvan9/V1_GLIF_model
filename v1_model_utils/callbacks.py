import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime as dt
from time import time
import pickle as pkl
from matplotlib import pyplot as plt
import seaborn as sns
# from scipy.signal import correlate
import stim_dataset
from v1_model_utils import other_v1_utils
from v1_model_utils.plotting_utils import InputActivityFigure, PopulationActivity
from v1_model_utils.model_metrics_analysis import ModelMetricsAnalysis


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
        _acc, _loss, _rate, _rate_loss, _voltage_loss, _osi_dsi_loss = metrics_values
        _s = f'Loss {_loss:.4f}, '
        _s += f'RLoss {_rate_loss:.4f}, '
        _s += f'VLoss {_voltage_loss:.4f}, '
        _s += f'OLoss {_osi_dsi_loss:.4f}, '
        _s += f'Accuracy {_acc:.4f}, '
        _s += f'Rate {_rate:.4f}'
        return _s


class Callbacks:
    def __init__(self, network, lgn_input, bkg_input, model, optimizer, distributed_roll_out, flags, logdir, strategy, 
                metrics_keys, pre_delay=50, post_delay=50, checkpoint=None, model_variables_init=None, 
                save_optimizer=True):
        
        self.n_neurons = flags.neurons
        self.network = network
        self.lgn_input = lgn_input
        self.bkg_input = bkg_input
        self.neuropixels_feature = 'Ave_Rate(Hz)'  
        self.model = model
        self.optimizer = optimizer
        self.flags = flags
        self.logdir = logdir
        self.strategy = strategy
        self.distributed_roll_out = distributed_roll_out
        self.metrics_keys = metrics_keys
        self.pre_delay = pre_delay
        self.post_delay = post_delay
        self.step = 0
        self.total_epochs = flags.n_runs * flags.n_epochs
        self.step_running_time = []
        self.model_variables_dict = model_variables_init
        self.initial_metric_values = None
        self.summary_writer = tf.summary.create_file_writer(self.logdir)
        with open(os.path.join(self.logdir, 'config.json'), 'w') as f:
            json.dump(flags.flag_values_dict(), f, indent=4)

        if checkpoint is None:
            if save_optimizer:
                checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
            else:
                checkpoint = tf.train.Checkpoint(model=model)
            self.min_val_loss = float('inf')
            self.no_improve_epochs = 0
            # create a dictionary to save the values of the metric keys after each epoch
            self.epoch_metric_values = {key: [] for key in self.metrics_keys}
        else:
            # Load epoch_metric_values and min_val_loss from the file
            try:
                with open(os.path.join(self.logdir, 'train_end_data.pkl'), 'rb') as f:
                    data_loaded = pkl.load(f)
                self.epoch_metric_values = data_loaded['epoch_metric_values']
                self.min_val_loss = data_loaded['min_val_loss']
                self.no_improve_epochs = data_loaded['no_improve_epochs']
            except FileNotFoundError:
                print('No train_end_data.pkl file found. Initializing...')
                self.min_val_loss = float('inf')
                self.no_improve_epochs = 0
                self.epoch_metric_values = {key: [] for key in self.metrics_keys}

        # Manager for the best model
        self.best_manager = tf.train.CheckpointManager(
            checkpoint, directory=self.logdir + '/Best_model', max_to_keep=1
        )
        self.latest_manager = tf.train.CheckpointManager(
            checkpoint, directory=self.logdir + '/latest', max_to_keep=1
        )
        # Manager for osi/dsi checkpoints 
        self.epoch_manager = tf.train.CheckpointManager(
            checkpoint, directory=self.logdir + '/OSI_DSI_checkpoints', max_to_keep=None
        )


    def on_train_begin(self):
        print("---------- Training started at ", dt.datetime.now().strftime('%d-%m-%Y %H:%M'), ' ----------\n')
        self.train_start_time = time()
        self.epoch = self.flags.run_session * self.flags.n_epochs

    def on_train_end(self, metric_values, normalizers=None):
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

        # Save epoch_metric_values and min_val_loss to a file
        data_to_save = {
            'epoch_metric_values': self.epoch_metric_values,
            'min_val_loss': self.min_val_loss,
            'no_improve_epochs': self.no_improve_epochs
        }
        if normalizers is not None:
            data_to_save['v1_ema'] = normalizers['v1_ema']

        with open(os.path.join(self.logdir, 'train_end_data.pkl'), 'wb') as f:
            pkl.dump(data_to_save, f)

        if self.flags.n_runs > 1:
            self.plot_osi_dsi(parallel=True)

    def on_epoch_start(self):
        self.epoch += 1
        # self.step_counter.assign_add(1)
        self.epoch_init_time = time()
        date_str = dt.datetime.now().strftime('%d-%m-%Y %H:%M')
        print(f'Epoch {self.epoch:2d}/{self.total_epochs} @ {date_str}')
        tf.print(f'\nEpoch {self.epoch:2d}/{self.total_epochs} @ {date_str}')

    def on_epoch_end(self, x, z, y, metric_values, verbose=True):
        self.step = 0
        if self.initial_metric_values is None:
            self.initial_metric_values = metric_values
        
        if verbose:
            print_str = f'  Validation: - Angle: {y[0][0]:.2f}\n' 
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
        
        # save latest model every 10 epochs
        if self.epoch % 10 == 0:
            self.save_latest_model()    

        if val_loss_value < self.min_val_loss:
        # if True:
            self.min_val_loss = val_loss_value
            self.no_improve_epochs = 0
            # self.plot_lgn_activity(x)
            self.save_best_model()
            self.plot_raster(x, z, y)
            self.plot_mean_firing_rate_boxplot(z, y)
            # self.plot_populations_activity(z)

            self.model_variables_dict['Best'] = {var.name: var.numpy() for var in self.model.trainable_variables}
            for var in self.model_variables_dict['Best'].keys():
                t0 = time()
                self.variable_change_analysis(var)
                print(f'Time spent in {var}: {time()-t0}')
        else:
            self.no_improve_epochs += 1

        # # Plot osi_dsi if only 1 run and the osi/dsi period is reached
        # if self.flags.n_runs == 1 and (self.epoch % self.flags.osi_dsi_eval_period == 0 or self.epoch==1):
        #     self.plot_osi_dsi(parallel=False)
           
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
        # reset the gpu memory stat
        tf.config.experimental.reset_memory_stats('GPU:0')

    def on_step_end(self, train_values, y, verbose=True):
        self.step_running_time.append(time() - self.step_init_time)
        if verbose:
            print_str = f'  Step {self.step:2d}/{self.flags.steps_per_epoch} - Angle: {y[0][0]:.2f}\n'
            print_str += '    ' + compose_str(train_values)
            print(print_str)
            print(f'    Step running time: {time() - self.step_init_time:.2f}s')
            mem_data = printgpu(verbose=1)
            print(f'    Memory consumption (current - peak): {mem_data[0]:.2f} GB - {mem_data[1]:.2f} GB')
        
    def save_latest_model(self):
        try:
            p = self.latest_manager.save(checkpoint_number=self.epoch)
            print(f'Latest model saved in {p}\n')    
        except:
            print("Saving failed. Maybe next time?")    

    def save_best_model(self):
        # self.step_counter.assign_add(1)
        print(f'[ Saving the model at epoch {self.epoch} ]')
        try:
            p = self.best_manager.save(checkpoint_number=self.epoch)
            print(f'Model saved in {p}\n')
        except:
            print("Saving failed. Maybe next time?")

    def plot_losses_curves(self):
        # plotting_metrics = ['val_loss', 'val_firing_rate', 'val_rate_loss', 'val_voltage_loss']
        plotting_metrics = ['val_loss', 'val_osi_dsi_loss', 'val_rate_loss', 'val_voltage_loss']
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
    
    def plot_raster(self, x, z, y):
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
                                    core_radius=self.flags.plot_core_radius,
                                    )
        graph(x, z)

    def plot_lgn_activity(self, x):
        x = x.numpy()[0, :, :]
        x_mean = np.mean(x, axis=1)
        plt.figure(figsize=(10, 5))
        plt.plot(x_mean)
        plt.title('Mean input activity')
        plt.xlabel('Time (ms)')
        plt.ylabel('Mean input activity')
        os.makedirs(os.path.join(self.logdir, 'Populations activity'), exist_ok=True)
        plt.savefig(os.path.join(self.logdir, 'Populations activity', f'LGN_population_activity_epoch_{self.epoch}.png'))

    def plot_populations_activity(self, z):
        z = z.numpy()

        # Plot the mean firing rate of the population of neurons
        filename = f'Epoch_{self.epoch}'
        Population_activity = PopulationActivity(n_neurons=self.n_neurons, network=self.network, 
                                                stimuli_init_time=self.pre_delay, stimuli_end_time=self.flags.seq_len-self.post_delay, 
                                                image_path=self.logdir, filename=filename, data_dir=self.flags.data_dir,
                                                core_radius=self.flags.plot_core_radius)
        Population_activity(z, plot_core_only=True, bin_size=10)


    def plot_mean_firing_rate_boxplot(self, z, y):
        z = z.numpy()
        y = y.numpy()
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{self.neuropixels_feature}')
        os.makedirs(boxplots_dir, exist_ok=True)
        metrics_analysis = ModelMetricsAnalysis(self.network, data_dir=self.flags.data_dir, n_trials=1,
                                                core_radius=self.flags.plot_core_radius, drifting_gratings_init=self.pre_delay, drifting_gratings_end=self.flags.seq_len-self.post_delay, 
                                                )
        metrics_analysis(z, y, metrics=[self.neuropixels_feature], directory=boxplots_dir, filename=f'Epoch_{self.epoch}')    


    def variable_change_analysis(self, variable):
        if 'rest_of_brain_weights' in variable:
            self.node_to_pop_weights_analysis(self.bkg_input['indices'], variable=variable)
        elif 'sparse_input_weights' in variable:
            self.node_to_pop_weights_analysis(self.lgn_input['indices'], variable=variable)
        elif 'sparse_recurrent_weights' in variable:
            self.pop_to_pop_weights_analysis(self.network['synapses']['indices'], variable=variable)
        

    def node_to_pop_weights_analysis(self, indices, variable=''):
        pop_names = other_v1_utils.pop_names(self.network)
        target_cell_types = [other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in pop_names]
        if 'rest_of_brain_weights' in variable:
            post_indices =  np.repeat(indices[:, 0], 4)
        else:
            post_indices = indices[:, 0]

        post_cell_types = [target_cell_types[i] for i in post_indices]
        # Create DataFrame with all the necessary data
        df = pd.DataFrame({
            'Cell type': post_cell_types * 2,  # Duplicate node names for initial and final weights
            'Weight': self.model_variables_dict['Initial'][variable].tolist() + self.model_variables_dict['Best'][variable].tolist(),  # Combine initial and final weights
            'State': ['Initial'] * len(self.model_variables_dict['Initial'][variable]) + ['Final'] * len(self.model_variables_dict['Best'][variable])  # Distinguish between initial and final weights
        })

        # Sort the dataframe by Node Name and then by Type to ensure consistent order
        df = df.sort_values(['Cell type', 'State'])

        # Plotting
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{variable}')
        os.makedirs(boxplots_dir, exist_ok=True)
        fig, axs = plt.subplots(2, 1, figsize=(12, 14))

        fig = plt.figure(figsize=(12, 6))
        hue_order = ['Initial', 'Final']
        # sns.boxplot(x='Node Name', y='Weight Change', data=df)
        sns.barplot(x='Cell type', y='Weight', hue='State', hue_order=hue_order, data=df)
        # sns.boxplot(x='Node Name', y='Weight', hue='Type', hue_order=hue_order, data=df)
        # plt.axhline(0, color='black', linewidth=1)  # include a horizontal black line at 0
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
        plt.title(f'{variable}')
        plt.tight_layout()
        plt.savefig(os.path.join(boxplots_dir, f'{variable}.png'), dpi=300, transparent=False)
        plt.close()

    def pop_to_pop_weights_analysis(self, indices, variable=''):
        pop_names = other_v1_utils.pop_names(self.network)
        cell_types = [other_v1_utils.pop_name_to_cell_type(pop_name) for pop_name in pop_names]
        post_cell_types = [cell_types[i] for i in indices[:, 0]]
        pre_cell_types = [cell_types[i] for i in indices[:, 1]]

        ### Initial Weight ###
        weight_changes = self.model_variables_dict['Best'][variable] - self.model_variables_dict['Initial'][variable]
        df = pd.DataFrame({'Post Name': post_cell_types, 'Pre_names':pre_cell_types, 
                            'Initial weight': self.model_variables_dict['Initial'][variable], 'Final weight': self.model_variables_dict['Best'][variable], 
                            'Weight Change': weight_changes})
        
        # Calculate global min and max for color normalization
        global_grouped_df = df.groupby(['Pre_names', 'Post Name'])[['Initial weight', 'Final weight']].mean().reset_index()
        global_min = global_grouped_df[['Initial weight', 'Final weight']].min().min()
        global_max = global_grouped_df[['Initial weight', 'Final weight']].max().max()

        # Plot for Initial Weight
        grouped_df = df.groupby(['Pre_names', 'Post Name'])['Initial weight'].mean().reset_index()
        # Create a pivot table to reshape the data for the heatmap
        pivot_df = grouped_df.pivot(index='Pre_names', columns='Post Name', values='Initial weight')
        # Plot heatmap
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{variable}')
        os.makedirs(boxplots_dir, exist_ok=True)
        fig = plt.figure(figsize=(12, 6))
        heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=global_min, vmax=global_max)
        # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-0.5, vmax=0.5)
        plt.xticks(rotation=90)
        plt.gca().set_aspect('equal')
        plt.title(f'{variable}')
        # Create a separate color bar axis
        cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
        # Plot color bar
        cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
        cbar.set_label('Initial Weight')
        plt.savefig(os.path.join(boxplots_dir, f'Initial_weight.png'), dpi=300, transparent=False)
        plt.close()

        ### Final Weight ###
        grouped_df = df.groupby(['Pre_names', 'Post Name'])['Final weight'].mean().reset_index()
        # Create a pivot table to reshape the data for the heatmap
        pivot_df = grouped_df.pivot(index='Pre_names', columns='Post Name', values='Final weight')
        # Plot heatmap
        plt.figure(figsize=(12, 6))
        heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=global_min, vmax=global_max)
        # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-0.5, vmax=0.5)
        plt.xticks(rotation=90)
        plt.gca().set_aspect('equal')
        plt.title(f'{variable}')
        # Create a separate color bar axis
        cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
        # Plot color bar
        cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
        cbar.set_label('Final Weight')
        plt.savefig(os.path.join(boxplots_dir, f'Final_weight.png'), dpi=300, transparent=False)
        plt.close()

        ### Weight change ###
        grouped_df = df.groupby(['Pre_names', 'Post Name'])['Weight Change'].mean().reset_index()
        # Create a pivot table to reshape the data for the heatmap
        pivot_df = grouped_df.pivot(index='Pre_names', columns='Post Name', values='Weight Change')
        # Plot heatmap
        plt.figure(figsize=(12, 6))
        heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=global_min, vmax=global_max)
        # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False)
        plt.xticks(rotation=90)
        plt.gca().set_aspect('equal')
        plt.title(f'{variable}')
        # Create a separate color bar axis
        cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
        # Plot color bar
        cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
        cbar.set_label('Weight Change')
        plt.savefig(os.path.join(boxplots_dir, f'Weight_change.png'), dpi=300, transparent=False)
        plt.close()



    def plot_osi_dsi(self, parallel=False):
        print('Starting to plot OSI and DSI...')
        # Save the checkpoint to reload weights in the osi_dsi_estimator
        if parallel:
            p = self.epoch_manager.save(checkpoint_number=self.epoch)
            print(f'Checkpoint model saved in {p}\n')
        else:              
             # osi_dsi_data_set = self.strategy.distribute_datasets_from_function(self.get_osi_dsi_dataset_fn(regular=True))
            DG_angles = np.arange(0, 360, 45)
            osi_dataset_path = os.path.join('OSI_DSI_dataset', 'lgn_firing_rates.pkl')
            if not os.path.exists(osi_dataset_path):
                print('Creating OSI/DSI dataset...')
                # Define OSI/DSI dataset
                def get_osi_dsi_dataset_fn(regular=False):
                    def _f(input_context):
                        post_delay = self.flags.seq_len - (2500 % self.flags.seq_len)
                        _lgn_firing_rates = stim_dataset.generate_drifting_grating_tuning(
                            seq_len=2500+post_delay,
                            pre_delay=500,
                            post_delay = post_delay,
                            n_input=self.flags.n_input,
                            regular=regular,
                            return_firing_rates=True,
                            rotation=self.flags.rotation,
                        ).batch(1)
                                    
                        return _lgn_firing_rates
                    return _f
            
                osi_dsi_data_set = self.strategy.distribute_datasets_from_function(get_osi_dsi_dataset_fn(regular=True))
                test_it = iter(osi_dsi_data_set)
                lgn_firing_rates_dict = {}  # Dictionary to store firing rates
                for angle_id, angle in enumerate(DG_angles):
                    t0 = time()
                    lgn_firing_rates = next(test_it)
                    lgn_firing_rates_dict[angle] = lgn_firing_rates.numpy()
                    print(f'Angle {angle} done.')
                    print(f'    Trial running time: {time() - t0:.2f}s')
                    mem_data = printgpu(verbose=1)
                    print(f'    Memory consumption (current - peak): {mem_data[0]:.2f} GB - {mem_data[1]:.2f} GB\n')

                # Save the dataset      
                results_dir = os.path.join("OSI_DSI_dataset")
                os.makedirs(results_dir, exist_ok=True)
                with open(osi_dataset_path, 'wb') as f:
                    pkl.dump(lgn_firing_rates_dict, f)
                print('OSI/DSI dataset created successfully!')

            else:
                # Load the LGN firing rates dataset
                with open(osi_dataset_path, 'rb') as f:
                    lgn_firing_rates_dict = pkl.load(f)

            sim_duration = (2500//self.flags.seq_len + 1) * self.flags.seq_len
            n_trials_per_angle = 10
            spikes = np.zeros((8, sim_duration, self.flags.neurons), dtype=float)
            DG_angles = np.arange(0, 360, 45)

            for angle_id, angle in enumerate(range(0, 360, 45)):
                # load LGN firign rates for the given angle and calculate spiking probability
                lgn_fr = lgn_firing_rates_dict[angle]
                lgn_fr = tf.constant(lgn_fr, dtype=tf.float32)
                _p = 1 - tf.exp(-lgn_fr / 1000.)

                # test_it = iter(osi_dsi_data_set)
                for trial_id in range(n_trials_per_angle):
                    t0 = time()
                    # Reset the memory stats
                    tf.config.experimental.reset_memory_stats('GPU:0')
                    # Generate LGN spikes
                    x = tf.random.uniform(tf.shape(_p)) < _p
                    y = tf.constant(angle, dtype=tf.float32, shape=(1,1))
                    w = tf.constant(sim_duration, dtype=tf.float32, shape=(1,))

                    # x, y, _, w = next(test_it)
                    chunk_size = self.flags.seq_len
                    num_chunks = (2500//chunk_size + 1)
                    for i in range(num_chunks):
                        chunk = x[:, i * chunk_size : (i + 1) * chunk_size, :]
                        z_chunk = self.distributed_roll_out(chunk, y, w, output_spikes=True)
                        spikes[angle_id, i * chunk_size : (i + 1) * chunk_size, :] += z_chunk.numpy()[0, :, :].astype(float)

                    if trial_id == 0 and angle_id == 0:
                        # Raster plot for 0 degree orientation
                        lgn_spikes = x[:, :2500, :].numpy()
                        z = spikes[:, :2500, :]
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
                                        core_radius=self.flags.plot_core_radius,
                                        )
                        graph(lgn_spikes, z)

                    print(f'Trial {trial_id}/{n_trials_per_angle} - Angle {angle} done.')
                    print(f'    Trial running time: {time() - t0:.2f}s')
                    mem_data = printgpu(verbose=1)
                    print(f'    Memory consumption (current - peak): {mem_data[0]:.2f} GB - {mem_data[1]:.2f} GB\n')
        
        	# Average the spikes over the number of trials
            spikes = spikes/n_trials_per_angle
            # Slice only the first 2500 ms
            spikes = spikes[:, :2500, :]

            # Do the OSI/DSI analysis 
            boxplots_dir = os.path.join(self.logdir, 'Boxplots_OSI_DSI')
            os.makedirs(boxplots_dir, exist_ok=True)
            metrics_analysis = ModelMetricsAnalysis(self.network, data_dir=self.flags.data_dir,
                                                    drifting_gratings_init=500, drifting_gratings_end=2500,
                                                    core_radius=self.flags.plot_core_radius)
            metrics_analysis(spikes, DG_angles, metrics=["Rate at preferred direction (Hz)", "OSI", "DSI"], directory=boxplots_dir, filename=f'Epoch_{self.epoch}')
