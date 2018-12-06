# AO-PWC

## Training
A predictive wavefront control network can be trained by running the `scripts/train.py` script. If the optional `--name` argument is provided, this will create a new results directory in the `./experiments` folder with config information, training results and experiment checkpoints. For example
```
python scripts/train.py --name my-exp
```
will create a new experiment at `./experiments/my-exp`. If the `--name` argument is omitted, the experiment will run but no results will be saved (handy for testing!). See the usage information below for details of additional command line arguments. 
```
usage: train.py [-h] [--name NAME] [--logdir LOGDIR] [--batch-size BATCH_SIZE]
                [--lr LR] [--epochs EPOCHS] [--steps-ahead STEPS_AHEAD]
                [--gpu GPU] [--arch {ConvLSTM}] [--hidden HIDDEN [HIDDEN ...]]
                [--workers WORKERS] [--train-split TRAIN_SPLIT]
                [--val-split VAL_SPLIT]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME, -n NAME  name of the experiment. If left blank, no logging
                        information will be saved to disk
  --logdir LOGDIR, -d LOGDIR
                        location to store experiment log files
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        number of examples per mini-batch
  --lr LR, -l LR        learning rate
  --epochs EPOCHS, -e EPOCHS
                        number of training epochs
  --steps-ahead STEPS_AHEAD, -s STEPS_AHEAD
                        number of timesteps into the future to predict
  --gpu GPU, -g GPU     index of current gpu (use -1 for cpu training)
  --arch {ConvLSTM}, -a {ConvLSTM}
                        name of model architecture
  --hidden HIDDEN [HIDDEN ...]
                        number of feature channels in hidden layers
  --workers WORKERS, -w WORKERS
                        number of worker threads for data loading (use 0 for
                        single-threaded mode)
  --train-split TRAIN_SPLIT
                        fraction of data used for training
  --val-split VAL_SPLIT
                        fraction of data used for validation
```

## Directory structure
An example of how we could structure the project to keep everything nice and organized, largely inspired by [this project](https://drivendata.github.io/cookiecutter-data-science/). This was just intended as a starting point so feel free to change! 
```
ao-pwc/
|-- data/               # Raw data is saved to here (but not checked into git)
|-- experiments/        # Experiment log files (also ignored from git)
|-- notebooks/          # Jupyter notebooks
|-- results/            # Final results, visualizations
|-- aopwc/              # All of the python source files go in here
    |-- data/           # Dataloaders etc.
    |-- models/         # Pytorch models and layers
    |-- evaluation/     # Evaluation metrics
    |-- visualization   # Functions for visualizing outputs
    |-- utils/          # Helper functions
    |-- __init__.py
|-- scripts/            # Python/bash scripts for training, preprocessing data etc.
|-- tests/              # Unit tests (if we need them)
|-- requirements.txt    # Required python packages for virtual enviroment 
                        # (run `pip freeze > requirements.txt` to generate)
|-- README.md
|-- .gitignore
```

## Resources
- [**Guyon & Males (2017)**](https://arxiv.org/pdf/1707.00570.pdf) - Baseline method using linear algebra approach.
- [**PredNet**](https://coxlab.github.io/prednet/) - A popular LSTM-based model for video prediction from the paper "Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning". I was thinking this might be a good architecture to start from, maybe alongside a few more simple baselines.
- [**Deep multi-scale video prediction beyond mean square error**](https://arxiv.org/pdf/1511.05440.pdf) - Convolutional video prediction network from Yann LeCun and Facebook AI.
- [**Understanding LSTM Networks**](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) - An excellent blog post explaining the basics of LSTMs and recurrent neural networks.