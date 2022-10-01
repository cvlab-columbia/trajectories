# Configuration

This project uses [Hydra](https://hydra.cc/docs/intro/) for configuration. This README is divided in four sections.
First, project-specific configurations. Second, Hydra general configuration details that are important for this project.
Third, how to run an experiment with the correct configuration. And finally, a section explaining some of the
parameters.

## Project-specific

Under the current `config` folder there are several other directories and a `default.yaml` file. All the directories
except for `experiments` and `debug` represent config groups. The `default.yaml`, as the name says, contains all the
default configs, and calls default configurations for each config group.

`experiments` contains the actual experiments that we run, and are organized by type of experiment. This structure is
not related to any config group, so it can be broken or changed, it is just to organize the files. Each experiment first
calls whole config groups that need to change with respect to the `default.yaml`
configuration, and then changes other attributes specifically.

When creating a new config file in `experiments`, if you copy paste a previous one, make sure to change the
*wandb.name* attribute, othewise it can be confusing afterwards.

The configurations in `debug` follow the same rules as the `experiments` one, and are designed to be (potentially) added
after the experiments config, as a modification (see [_Run experiments_](#run-experiments]))

## General Hydra

All the documentation about Hydra is in their docs, but here we highlight some details that are important or tricky, at
least for this project.

- **About overwriting**. The following two points will apply silently, this is why they are important to have in mind.
    - Any parameter that is in `default.yaml` that is NOT in the *defaults* will not be able to be overwritten. This is
      why the `default.yaml` only contains items in the defaults.
    - Similarly, imports in the *defaults* that are not importing a specific config group will not be able to be
      overwritten (see [this](https://hydra.cc/docs/0.11/tutorial/defaults/))

To add a new attribute from the command line, use `++` (or no prefix), to modify use `+`, to remove use `~`.

The idea is that we pass all the config group all the way to the class definition. Very clean. Potential problems:

1) If we create a new attribute that may be useful for a specific experiment, make sure to either add a default in the
   class init for the experiments that do not have that attribute defined, or to add a default in the default config
   file.

## Run experiments

The information here is a small extension to the information in the main README.

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

To debug, we have to add a debug configuration from the `configs/debug` folder, like  `debug/debug.yaml`. It is
recommended to add it after the experiment, in which case it will overwrite any conflicting parameter defined in the
experiment. If added before, any conflicting parameter will be the one from the experiment_name:

```bash
python run.py +experiments=experiment_name +debug=debug   # after (recommended)
python run.py +debug=debug +experiments=experiment_name   # before
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

If for whatever reason you want to combine the parameters of two experiments (for example if you define the `debug.yaml`
as an experiment), you combine them like this:

```bash
python run.py +experiments=[experiment_name_1,experiment_name_2]
```

Make sure you control which one is overwriting the other in case there are incompatibilities.

## Important parameters (possibilities)

Other than optimization parameters or dataset parameters. They are described in more detail in the corresponding parts
of the code.

- `latent_distribution`. Latent-space family distribution.
- `distance_type`. Input-space distance.
- `loss_type` in `trajectory_loss`. Triplet, contrastive, or BCE.
- `generate_extrapolation`, `reconstruct_intersection`, `reencode`, `use_all`. Control the segments that are encoded and
  re-encoded.
- `symmetric_dist`. Differentiates between the symmetric (`True`) or conditional (`False`) approaches.
- `option_reencode`. How to combine representations from different samples of the same distribution.

Other parameters I may want to change:

- `time_encoder_strategy`, `time_decoder_strategy`. Fourier time encodings or MLP time encodings.
- `hidden_size`, `feature_size`
- `num_layers`
- `loss_type` in `reconstruction_loss`, although regression makes the most sense.

### Some prior information

There are 6 main config groups, represented by folders under `config`: _checkpoint, dataset, model, resume, trainer,
wandb_. The parameters defined in each of these configs will be sent separately to different parts of the code. For
example, when creating the model, the class defining the model will be initialized using the parameters from the
_model_ config group.

Each config group can have sub-config groups, that can change depending on other parameters of the config group. For
example, if `model.name=TwoBranches` (this is not a case in our code, just an example), we may have a sub-config group
called `model.model_params.params_branch_two` with attributes such as
`model.model_params.params_branch_two.num_layers=2`. However, if `model.name=OneBranch`, then that config group will not
even exist in the config file.

This (the fact that sub-config groups can change) is one of the reason why we do not mention __all__ the parameters here
in the README. Another reason is that a lot of parameters are default values in the python methods, and most of the time
they are not even mentioned in the config. **The parameters are best explained in comments in the context of the method
they are input to**. The config structure is flexible enough to change them in case it is needed. For example, let's say
we have a method that initializes an object:

```python
def create_example(param_1, param_2='default_2', param_3='default_3'):
    do_something()
```

Let's assume we have a configuration that looks like the following:

```yaml
example_group:
  param_1: 'a'
  param_2: 'b'
```

Then from the main python method (in our case `run.py`) we could call the function like this:

```python
create_example(**config.example_group)
```

Here, the value for `param_3` would be the default "default_3", which we did not even add to the config.

### Now finally, the parameters

Some of the configuration parameters, to get an idea of the structure of the configuration:

```yaml
setting: [ train/test/predict ]           # Run setting (type of experiment to run)
checkpoint:
  save_top_k: [ int, e.g. 3 ]             # Number of checkpoints to save (with best accuracy)
dataset:
  dataset_name: MyDataset1              # Dataset to use
  dataloader_params:
    num_workers: 8                      # Number of workers to load elements from the dataset
    batch_size: 16                      # Batch size
  split_use: [ train/validate/test/all ]  # Dataset split to use for experiment. Only used at test time
  dataset_params:
    # Specific dataset parameters will depend on the dataset
    max_steps: 10                       # Maximum number of steps sampled from a trajectory
    min_steps_past: 3                   # Minimum number of steps sampled in the past segment
    min_steps_future: 3                 # Minimum number of steps sampled in the future segment
model:
  name: TrajectoryModel                 # Model class to be used
  time_encoder_strategy: 'fourier'      # How time is represented in the encoder input
  time_decoder_strategy: 'fourier'      # How time is represented in the decoder input
  latent_distribution: 'bbox'           # Distribution family
  point_trajectory: False               # If True, segments are represented as points, not distributions
  num_sample_points: 3                  # Number of points to be sampled from a distribution before decoding them
  distance_type: 'prediction'           # Distance function between distributions
  hidden_size: 512                      # Size of the inner model activations
  feature_size: 512                     # Size of the representation
  option_reencode: 2                    # Re-encoding can be done in different ways
  optim_params:
    base_lr: [ float, e.g. 0.0001 ]                 # Base learning rate. Usage will depend on the lr_policy
    lr_policy: [ cosine,steps_with_relative_lrs ]   # Policy to modify learning rate during training
    optimizing_method: [ sgd,adam,adamw ]           # Optimizer to be used
  loss_params:
    dict_losses:
      contrastive:
        λ: 1
        params:
  generate_extrapolation: True          # Predict points in segments that are not part of the input
  reconstruct_intersection: True        # Compute intersection of past and future, and decode from it
  reencode: True                        # Use the re-encoding strategy
  use_all: True                         # Use the "all" segment ("combination" in the paper)
  symmetric_dist: False
resume:
  id: [ e.g. 1234abcd ]           # ID of experiment to resume from
  path: path_to_checkpoint      # Path to checkpoint to resume from
  load_all: [ true/false ]        # Load everything from the previous model. Implies continuing the experiment (if true)
  load_state: [ true/false ]      # Start new experiment, only load model weights and optimization state (if true)
  load_model: [ true/false ]      # Start new experiment, only load model weights (if true)
  check_config: [ true/false ]    # Check that config from loaded model is the same as current config
trainer:
  precision: [ 16,32 ]            # 16 is half precision, 32 is regular float precision      
  gpus: 1                       # Number of GPUs to use. To specifiy the IDs, run with CUDA_VISIBLE_DEVICES=X,Y,Z
wandb:
  name: name_of_experiment      # Name that will show in wandb. Ideally, set the same name as the name of the yaml file
  save_dir: path_save           # Path to logs and checkpoints
```