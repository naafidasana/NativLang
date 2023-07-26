import torch
import numpy as np

import matplotlib.pyplot as plt
from IPython import display

import requests
import os
import tarfile
import zipfile
import hashlib

import time


def get_gpus():
    """Use all GPUs if there is at least on GPU on the system else CPU."""
    devices = [torch.device(f"cuda:{i}")
               for i in range(torch.cuda.device_count())]

    return (devices, torch.cuda.device_count()) if devices else ([torch.device("cpu")], 0)


def setaxes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    if legend:
        axes.legend(legend)
    axes.grid()


HUB = dict()
URL = "https://zenodo.org/record/8186835/files/{file}"

# Add items to HUB
HUB["dag-sents-train"] = (URL.format(file="dag-sents-train.zip"),
                         "456e9cc93e70b79110e7421475z9bcd8d39218ce")


def download(name, cache_dir=os.path.join('.', 'data')):
    assert name in HUB, f"{name} does not exist in {HUB}."
    url, sha1_hash = HUB[name]
    # If downloaded already, load from disk
    filename = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(filename):
        sha1 = hashlib.sha1()
        with open(filename, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return filename

    # Download from url
    print(f"Downloading {filename} from {url}...")
    response = requests.get(url, stream=True, verify=True)
    with open(filename, "wb") as f:
        f.write(response.content)

    return filename


def download_and_extract(name, destination_folder=None):
    filename = download(name)
    base_dir = os.path.dirname(filename)
    data_dir, ext = os.path.splitext(filename)
    if ext == '.zip':
        file = zipfile.ZipFile(filename, 'r')
    elif ext in ['.tar', '.gz']:
        file = tarfile.open(filename, 'r')
    else:
        assert False, 'Only .zip or .tar/.gz files can be extracted.'
    file.extractall(base_dir)
    return os.path.join(base_dir, destination_folder) if destination_folder else data_dir


class Visualizer:
    """For plotting data."""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'f:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Plot multiple lines incrementally
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        # Capture arguments with lamda function
        self.config_axes = lambda: setaxes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend
        )
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


class MetricAccumulator:
    """Accumulate metrics over `n` variables/steps."""

    def __init__(self, n):
        self.content = [0.0] * n

    def add(self, *args):
        self.content = [a + float(b) for a, b in zip(self.content, args)]

    def reset(self):
        self.content = [0.0]*len(self.content)

    def __getitem__(self, ndx):
        return self.content[ndx]


class Timer:
    """Keep track of multiple running times"""

    def __init__(self):
        self.times_elapsed = []
        self.start()

    def start(self):
        """Start the timer."""
        self.start_time = time.time()

    def stop(self):
        """End the timer."""
        self.times_elapsed.append(time.time() - self.start_time)
        return self.times_elapsed[-1]

    def sum(self):
        """Return the sum of elapsed times."""
        return sum(self.times_elapsed)

    def cumsum(self):
        """Return the accumulated times."""
        return np.array(self.times_elapsed).cumsum().tolist()

    def avg(self):
        """Return the average time elapsed."""
        return self.sum()/len(self.times_elapsed)

#filepath = download_and_extract("dag-sents-train")
#print(filepath)
