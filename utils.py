# from https://github.com/jaywalnut310/vits
import os
import sys
import logging
import subprocess
import torch
import numpy as np
from omegaconf import OmegaConf
from scipy.io.wavfile import read

MATPLOTLIB_FLAG = False

logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO, 
    format='[%(levelname)s|%(filename)s:%(lineno)s][%(asctime)s] >>> %(message)s'
)
logger = logging


def load_checkpoint(checkpoint_path, rank=0, model_g=None, model_d=None, optim_g=None, optim_d=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    config = checkpoint_dict['config']

    if model_g is not None:
        model_g, optim_g = load_model(
            model_g, 
            checkpoint_dict['model_g'],
            optim_g,
            checkpoint_dict['optimizer_g'])

    if model_d is not None:
        model_d, optim_d = load_model(
            model_d, 
            checkpoint_dict['model_d'],
            optim_d,
            checkpoint_dict['optimizer_d'])
    if rank == 0:
        logger.info(
            "Loaded checkpoint '{}' (iteration {})".format(
                checkpoint_path, 
                iteration
            )
        )
    return model_g, model_d, optim_g, optim_d, learning_rate, iteration, config
    
def load_checkpoint_diffsize(checkpoint_path, rank=0, model_g=None, model_d=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    config = checkpoint_dict['config']

    if model_g is not None:
        model_g = load_model_diffsize(
            model_g, 
            checkpoint_dict['model_g'])
    if model_d is not None:
        model_d = load_model_diffsize(
            model_d, 
            checkpoint_dict['model_d'])
    if rank == 0:
        logger.info(
            "Loaded checkpoint '{}' (iteration {})".format(
                checkpoint_path, 
                iteration
            )
        )
    del checkpoint_dict
    return model_g, model_d, learning_rate, iteration, config
 
def load_model_diffsize(model, model_state_dict):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    for k, v in model_state_dict.items():
        if k in state_dict and state_dict[k].size() == v.size():
            state_dict[k] = v
    
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
        
    return model



def load_model(model, model_state_dict, optim, optim_state_dict):
    if optim is not None:
        optim.load_state_dict(optim_state_dict)

    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    for k, v in model_state_dict.items():
        if k in state_dict and state_dict[k].size() == v.size():
            state_dict[k] = v
    
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
        
    return model, optim


def save_checkpoint(net_g, optim_g, net_d, optim_d, hps, epoch, learning_rate, save_path):
    
    def get_state_dict(model):
        if hasattr(model, 'module'):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        return state_dict
    
    torch.save({'model_g': get_state_dict(net_g),
                'model_d': get_state_dict(net_d),
                'optimizer_g': optim_g.state_dict(),
                'optimizer_d': optim_d.state_dict(),
                'config': str(hps),
                'iteration': epoch,
                'learning_rate': learning_rate}, save_path)


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats='HWC')
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path):
    sampling_rate, wav = read(full_path)
    
    if len(wav.shape) == 2:
        wav = wav[:, 0]

    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0
    wav = wav.astype(np.float32)
    return torch.FloatTensor(wav), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_hparams(args, init=True):
    config = OmegaConf.load(args.config)
    hparams = HParams(**config)
    model_dir = os.path.join(hparams.train.log_path, args.model)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    hparams.model_name = args.model
    hparams.model_dir = model_dir
    config_save_path = os.path.join(model_dir, "config.yaml")

    if init:
        OmegaConf.save(config, config_save_path)

    return hparams


def get_hparams_from_file(config_path):
    config = OmegaConf.load(config_path)
    hparams = HParams(**config)
    return hparams


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn("{} is not a git repository, therefore hash value comparison will be ignored.".format(
            source_dir
        ))
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn("git hash values are different. {}(saved) != {}(current)".format(
                saved_hash[:8], cur_hash[:8]))
    else:
        open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
