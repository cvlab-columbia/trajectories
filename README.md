# Trajectory Prediction

### Table of contents

1. [Run](#run)
2. [Runs](#runs)
3. [Data](#data)
4. [Requirements](#requirements)
5. [Code Structure](#code-structure)

## Run

This repository uses
[Pytorch Lightning](https://www.google.com/search?client=safari&rls=en&q=pytorch+lightning&ie=UTF-8&oe=UTF-8) to
implement the training and models, [Hydra](https://hydra.cc/docs/intro/) to define the configurations, and
[Wandb](https://wandb.ai/home) to visualize the training.

The experiments are defined as YAML files in the `configs/experiments` folder. For more detailed information on the
structure of the config files, and how to create them, read `configs/README.md`. Here we show the basic information.

To run an experiment, call the `run.py` file with the name of the experiment:

```bash
python run.py +experiments=experiment_name  # without the .yaml extension
```

If the experiment name is inside a subfolder of the `experiments` directory, simply add it:

```bash
python run.py +experiments=folder_experiment/experiment_name
```

A specific command to run an actual experiment in this project would be:

```bash
python run.py +experiments=train/train_finegym/finegym_bbox_triplet_asym
```

which trains our model with bounding box representations, for the short sequences case in FineGym. All the other
training configuration files are under the same directory.

To debug, we have to add a debug configuration from the `configs/debug` folder, like  `debug/debug.yaml`. It is
recommended to add it after the experiment:

```bash
python run.py +experiments=experiment_name +debug=debug 
```

For small changes that do not require creating a new experiment YAML file, we can just add the parameters in the same
command:

```bash
python run.py +experiments=experiment_name \
    wandb.name=new_name \
    dataset.dataloader_params.batch_size=8 \
    ++trainer.max_epochs=100
    +trainer.new_param_trainer=0.001 \
    ~trainer.profiler
```

where `~` removes a parameter from the configuration, `+` adds a non-existing parameter, and no prefix or `++` changes
an existing parameter.

### Resume and Pre-train

There are different options under `resume`:

- `load_all`. Resuming same training. In this case we want to load weights, training state, wandb run, and have same
  config for dataset and everything else. We explicitly make sure all the config is the same, including dataset,
  dataloader, etc. The `id` has to be defined.
- `load_state`. Pre-train from a previous checkpoint, load both training state and model.
- `load_model`. Pre-train from a previous checkpoint, only model.

The priority is from top to bottom. So if `load_all` is true and `load_state` is false, `load_all` prevails.

If any of the first three is set, either the `id` of the experiment we are loading from, or the `path` of the checkpoint
we are loading from have to be set.

For `load_state` and `load_model` we do not check configuration. If parameters like `lr` are set, they will be
overwritten, but other configurations like a different optimizer or different network size will break because the load
will not work.

The option of resuming training with different parameters (like learning rate), but under the same wandb run and model
folder is *not* supported, because it is confusing and bad for reproducibility. wandb logs in the filesystem (not on the
web application) each run separately even when they have the same id (which is good and clear), but still not enough for
this feature to be supported.

## Runs

The checkpoints and wandb logs are stored in the `wandb.save_dir` directory (under `{wandb.project}` and `wandb`
folders, respectively). The checkpoints are stored with the experiment id (e.g. _1234abcd_), and the logs under the run
ID, which has a format like _run-{date}\_{time}-{id}_. The run name is not necessary, it is loaded from the experiment
id.

The logs are also stored online, and can be accessed in https://wandb.ai (you will need to create an account). The
experiment ID can be found by accessing a specific run, going to overview ("info" sign top left), and check the "run
path". Change wandb config in `configs/wandb/wandb.yaml`

## Data

We use the FineGym (downloadable from [this link](https://sdolivia.github.io/FineGym/)), Diving48 (
in [this link](http://www.svcl.ucsd.edu/projects/resound/dataset.html)), and
FisV ([this link](https://drive.google.com/file/d/1FQ0-H3gkdlcoNiCe8RtAoZ3n7H1psVCI/view)) datasets.

The process to obtain the keypoints that the model uses is described next.

### Preprocess Data

There are three steps to obtain keypoints from data:

1. Extract keypoints from either videos or images using OpenPose. The code in `extract_keypoints_images.py` and
   `extract_keypoints_videos.py` under `data/data_utils` does that. We used OpenPose in a docker installation.
2. For videos that may contain multiple shots, extract divisions between shots in using `shot_detection.py`.
3. Post-process keypoints to group them into trajectories (they are initially extracted per-frame). This is done
   automatically during the dataset creation when running experiments.

## Requirements

The configuration of the cuda environment is in `requirements.yml`. To create an environment with the same packages,
run:

```bash
conda env create --file requirements.yml
```

## Code Structure

The code is divided in different files and folders (all python):

- `run.py`. Main file to be executed. Loads the configuration, creates model, trainer, and dataloader, and runs them.
- `losses.py`. File with loss functions and other evaluation functions.
- `distances.py`. File with distance functions.
- `data`. Dataset and dataloader code. Relies on a
  [LightningDataModule](https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html), defined in
  `main_data_module.py`, that manages the dataset. The datasets are defined under `data/datasets`, and all inherit from
  the `BaseDataset` defined in `base_dataset.py`. There is also a `data_utils` folder with general dataset utils.
- `models`. Under this folder we define the python modules (`nn.Module`), under `networks`, as well as the trainer,
  which is implemented using
  [LightningModule](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html). The lightning
  modules encapsulate all the training procedure, as well as the model definition. `trajectory_dict.py` is an auxiliary
  file that defines the state of all input- and latent-space trajectories.
- `utils`. General utils for the project.

Most of the files and methods are described in the code. For more specific comments about how they work and what they
do, go directly to the files.
