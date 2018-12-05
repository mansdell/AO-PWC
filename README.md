# AO-PWC

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