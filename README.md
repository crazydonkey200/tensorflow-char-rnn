# TensorFlow-Char-RNN
A TensorFlow implementation of Andrej Karpathy's [Char-RNN](https://github.com/karpathy/char-rnn), a character level language model using multilayer Recurrent Neural Network (RNN, LSTM or GRU). See his article [The Unreasonable Effectiveness of Recurrent Neural Network](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) to learn more about this model. 

# Installation

## Dependencies
- Python 2.7
- TensorFlow >= 1.2

Follow the instructions on [TensorFlow official website](https://www.tensorflow.org/) to install TensorFlow. 

# Test

If the installation finishes with no error, quickly test your installation by running:
```bash
python train.py --data_file=data/tiny_shakespeare.txt --num_epochs=10 --test
```

This will train char-rnn on the first 1000 characters of the tiny shakespeare copus. The final train/valid/test perplexity should all be lower than 30. 

# Usage
- `train.py` is the script for training.
- `sample.py` is the script for sampling.
- `char_rnn_model.py` implements the Char-RNN model.

## Training
To train on tiny shakespeare corpus (included in data/) with default settings (this might take a while):
```bash
python train.py --data_file=data/tiny_shakespeare.txt
```

All the output of this experiment will be saved in a folder (default to `output/`, you can specify the folder name using `--output_dir=your-output-folder`). 

The experiment log will be printed to stdout by default. To direct the log to a file instead, use `--log_to_file` (then it will be saved in `your-output-folder/experiment_log.txt`).

The output folder layout: 
```
  your-output-folder
    ├── result.json             # results (best validation and test perplexity) and experiment parameters.
    ├── vocab.json              # vocabulary extracted from the data.
    ├── experiment_log.txt      # Your experiment log if you used --log_to_file in training.
    ├── tensorboard_log         # Folder containing Logs for Tensorboard visualization.
    ├── best_model              # Folder containing saved best model (based on validation set perplexity)
    ├── saved_model             # Folder containing saved latest models (for continuing training).
```

Note: `train.py` assume the data file is using utf-8 encoding by default, use `--encoding=your-encoding` to specify the encoding if your data file cannot be decoded using utf-8.

## Sampling
To sample from the best model of an experiment (with a given start_text and length):
```bash
python sample.py --init_dir=your-output-folder --start_text="The meaning of life is" --length=100
```

## Visualization
To use Tensorboard (a visualization tool in TensorFlow) to [visualize the learning](https://www.tensorflow.org/get_started/summaries_and_tensorboard#tensorboard-visualizing-learning) (the "events" tab) and [the computation graph](https://www.tensorflow.org/versions/r0.8/how_tos/graph_viz/index.html#tensorboard-graph-visualization) (the "graph" tab).

First run:
```bash
tensorboard --logdir=your-output-folder/tensorboard_log
```

Then navigate your browser to [http://localhost:6006](http://localhost:6006) to view. You can also specify the port using `--port=your-port-number`. 

## Continuing an experiment
To continue a finished or interrupted experiment, run:
```bash
python train.py --data_file=your-data-file --init_dir=your-output-folder
```


## Hyperparameter tuning

`train.py` provides a list of hyperparameters you can tune.

To see the list of all hyperparameters, run:
```bash
python train.py --help
```
