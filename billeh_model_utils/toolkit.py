import os
import json
import numpy as np
import random
import string
import matplotlib.pyplot as plt


def exp_filter(_x, tau_n=5, n=5):
    l = int(tau_n * n)
    kernel = np.exp(-np.arange(l) / tau_n)
    kernel = kernel / np.sum(kernel)
    return np.convolve(_x, kernel)[:-l + 1]


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def apply_style(ax, scale=1, ylabel=.4):
    ax.set_xlabel(ax.get_xlabel(), fontsize=6 * scale)
    ax.set_ylabel(ax.get_ylabel(), fontsize=6 * scale)
    ax.spines['left'].set_linewidth(.5 * scale)
    ax.spines['bottom'].set_linewidth(.5 * scale)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=5 * scale, width=.5 * scale, length=3 * scale)
    ax.yaxis.set_tick_params(labelsize=5 * scale, width=.5 * scale, length=3 * scale)
    ax.yaxis.set_label_coords(-ylabel / 7, 0.5)


def do_inset_colorbar(_ax, _p, _label, loc='right'):
    if loc == 'right':
        bg_pos = [.925, .1, .075, .8]
        in_pos = [.95, .2, .025, .6]
    elif loc == 'left':
        bg_pos = [.025, .1, .15, .8]
        in_pos = [.05, .2, .025, .6]
    elif loc == 'middle':
        bg_pos = [.025 + .5, .1, .15, .8]
        in_pos = [.05 + .5, .2, .025, .6]
    else:
        raise NotImplementedError(f'must implement location {loc}')
    bg_ax = _ax.inset_axes(bg_pos)
    bg_ax.set_xticks([])
    bg_ax.set_yticks([])
    [_a.set_visible(False) for _a in bg_ax.spines.values()]
    inset_ax = _ax.inset_axes(in_pos)
    cbar = plt.colorbar(_p, cax=inset_ax)
    cbar.outline.set_visible(False)
    cbar.ax.set_ylabel(_label)


def get_random_identifier(prefix='', length=4):
    random_identifier = ''.join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(length))
    sim_name = prefix + random_identifier
    return sim_name

