# TensorFlow-Char-RNN
A TensorFlow implementation of Andrej Karpathy's [Char-RNN](https://github.com/karpathy/char-rnn), a character level language model using multilayer Recurrent Neural Network (RNN, LSTM or GRU). See his article [The Unreasonable Effectiveness of Recurrent Neural Network](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) to learn more about this model. 

# Installation

## Dependencies
- Python 2.7
- TensorFlow >= 0.7.0
- NumPy >= 1.10.0

Follow the instructions on [TensorFlow official website](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#download-and-setup) to install TensorFlow. 

I recommond using their pip installation:

```bash
# Ubuntu/Linux 64-bit, CPU only:
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled. Requires CUDA toolkit 7.5 and CuDNN v4.  For
# other versions, see "Install from sources" below.
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl

# Mac OS X, CPU only:
$ sudo easy_install --upgrade six
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0-py2-none-any.whl
```

It will install all the necessary packages for you. 

# Test

To quickly test your installation, run:
```bash
python char_rnn.py --data_file=data/tiny_shakespeare.txt --num_epochs=10 --test
```

This will train char-rnn on the first 1000 characters of the tiny shakespeare copus. The final train/valid/test perplexity should all be lower than 30. 

# Usage
`char_rnn.py` is the main script for training and sampling. 
`char_rnn_model.py` implements the Char-RNN model.

## Training
To train on tiny shakespeare corpus (included in data/) with default settings (this might take a while):
```bash
python char_rnn.py --data_file=data/tiny_shakespeare.txt
```

All the output of this experiment will be saved in a folder (default to output/, you can specify the folder name using `--output_dir=your-output-folder`). 

The experiment log will be printed to stdout by default. To direct the log to a file instead, use `--log_to_file` (then it will be saved in `your-output-folder/experiment_log.txt`).

The output folder layout: 
```
  your-output-folder
    ├── result.json             # results (best validation and test perplexity) and experiment parameters.
    ├── experiment_log.txt      # Your experiment log if you used --log_to_file in training.
    ├── tensorboard_log         # Folder containing Logs for Tensorboard visualization.
    ├── best_model              # Folder containing saved best model (based on validation set perplexity)
    ├── saved_model             # Folder containing saved latest models (for continuing training).
```

## Sampling
To sample from the best model of an experiment (with a given start_text and length):
```bash
python char_rnn.py --sample --init_from_dir=your-output-folder --start_text="The meaning of life is" --length=100
```

## Visualization
To use Tensorboard (a visualization tool in TensorFlow) to [visualize the learning] (https://www.tensorflow.org/versions/r0.8/how_tos/summaries_and_tensorboard/index.html#tensorboard-visualizing-learning) (the "events" tab) and [the computation graph](https://www.tensorflow.org/versions/r0.8/how_tos/graph_viz/index.html#tensorboard-graph-visualization) (the "graph" tab).

First run:
```bash
tensorboard --logdir=your-output-folder/tensorboard_log
```

Then navigate your browser to `localhost:6006` to view. You can also specify the port using `--port=your-port-number`. 

## Continue a finished or interrupted experiment
To continue a finished or interrupted experiment, run:
```bash
python char_rnn.py --data_file=your-data-file --init_from_dir=your-output-folder
```


## Hyperparameter tuning

`char_rnn.py` provides a list of hyperparameters you can tune.

To see the list of all hyperparameters, run:
```bash
python char_rnn.py --help
```

# License
Apache 2.0

