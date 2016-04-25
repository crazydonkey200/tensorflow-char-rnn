import argparse
import codecs
import json
import logging
import os
import shutil
import sys

import numpy as np
from char_rnn_model import *

TF_VERSION = int(tf.__version__.split('.')[1])

def main():
    parser = argparse.ArgumentParser()

    # Data and vocabulary file
    parser.add_argument('--data_file', type=str,
                        default='data/tiny_shakespeare.txt',
                        help='data file')

    parser.add_argument('--encoding', type=str,
                        default='utf-8',
                        help='the encoding of the data file.')

    # Parameters for saving models.
    parser.add_argument('--output_dir', type=str, default='output',
                        help=('directory to store final and'
                              ' intermediate results and models.'))

    # Parameters to configure the neural network.
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='size of RNN hidden state vector')
    parser.add_argument('--embedding_size', type=int, default=50,
                        help='size of character embeddings')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--num_unrollings', type=int, default=10,
                        help='number of unrolling steps.')

    # Parameters to control the training.
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='minibatch size')
    parser.add_argument('--train_frac', type=float, default=0.9,
                        help='fraction of data used for training.')
    parser.add_argument('--valid_frac', type=float, default=0.05,
                        help='fraction of data used for validation.')
    # test_frac is computed as (1 - train_frac - valid_frac).
    
    # Parameters for gradient descent.
    parser.add_argument('--max_grad_norm', type=float, default=5.,
                        help='clip global grad norm')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='initial learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')

    # Parameters for logging.
    parser.add_argument('--log_to_file', dest='log_to_file', action='store_true',
                        help=('whether the experiment log is stored in a file under'
                              '  output_dir or printed at stdout.'))
    parser.set_defaults(log_to_file=False)
    
    parser.add_argument('--progress_freq', type=int,
                        default=100,
                        help=('frequency for progress report in training'
                              ' and evalution.'))

    parser.add_argument('--verbose', type=int,
                        default=0,
                        help=('whether to show progress report in training'
                              ' and evalution.'))

    # Parameters to feed in the initial model and current best model.
    parser.add_argument('--init_model', type=str,
                        default='',
                        help=('initial model'))
    parser.add_argument('--best_model', type=str,
                        default='',
                        help=('current best model'))
    parser.add_argument('--best_valid_ppl', type=float,
                        default=np.Inf,
                        help=('current valid perplexity'))
    
    # Parameters for continuing training.
    parser.add_argument('--init_from_dir', type=str, default='',
                        help='continue from the outputs in the given directory')

    # Parameters for sampling.
    parser.add_argument('--sample', dest='sample', action='store_true',
                        help='sample new text that start with start_text')
    parser.set_defaults(sample=False)

    parser.add_argument('--max_prob', dest='max_prob', action='store_true',
                        help='always pick the most probable one in sampling')

    parser.set_defaults(max_prob=False)
    
    parser.add_argument('--start_text', type=str,
                        default='The meaning of life is ',
                        help='the text to start with')

    parser.add_argument('--length', type=int,
                        default=100,
                        help='length of sampled sequence')

    parser.add_argument('--seed', type=int,
                        default=-1,
                        help=('seed for sampling to replicate results, '
                              'an integer between 0 and 4294967295.'))

    # Parameters for evaluation (computing perplexity of given text).
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='compute the perplexity of given text')
    parser.set_defaults(evaluate=False)
    parser.add_argument('--example_text', type=str,
                        default='The meaning of life is 42.',
                        help='compute the perplexity of given example text.')

    # Parameters for debugging.
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='show debug information')
    parser.set_defaults(debug=False)

    # Parameters for unittesting the implementation.
    parser.add_argument('--test', dest='test', action='store_true',
                        help=('use the first 1000 character to as data'
                              ' to test the implementation'))
    parser.set_defaults(test=False)
    
    args = parser.parse_args()

    # Specifying location to store model, best model and tensorboard log.
    args.save_model = os.path.join(args.output_dir, 'save_model/model')
    args.save_best_model = os.path.join(args.output_dir, 'best_model/model')
    args.tb_log_dir = os.path.join(args.output_dir, 'tensorboard_log/')
    args.vocab_file = ''

    # Create necessary directories.
    if args.init_from_dir:
        args.output_dir = args.init_from_dir
    else:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        for paths in [args.save_model, args.save_best_model,
                      args.tb_log_dir]:
            os.makedirs(os.path.dirname(paths))

    # Specify logging config.
    if args.log_to_file:
        args.log_file = os.path.join(args.output_dir, 'experiment_log.txt')
    else:
        args.log_file = 'stdout'

    # Set logging file.
    if args.log_file == 'stdout':
        logging.basicConfig(stream=sys.stdout,
                            format='%(asctime)s %(levelname)s:%(message)s', 
                            level=logging.INFO,
                            datefmt='%I:%M:%S')
    else:
        logging.basicConfig(filename=args.log_file,
                            format='%(asctime)s %(levelname)s:%(message)s', 
                            level=logging.INFO,
                            datefmt='%I:%M:%S')

    print('all final and intermediate outputs will be stored in %s' % args.output_dir)
    print('all information will be logged to %s' % args.log_file)
    
    if args.debug:
        logging.info('args are:\n%s', args)

    # Prepare parameters.
    if args.init_from_dir:
        with open(os.path.join(args.init_from_dir, 'result.json'), 'r') as f:
            result = json.load(f)
        params = result['params']
        args.init_model = result['latest_model']
        best_model = result['best_model']
        best_valid_ppl = result['best_valid_ppl']
        args.encoding = result['encoding']
        args.vocab_file = os.path.join(args.init_from_dir, 'vocab.json')
    else:
        params = {'batch_size': args.batch_size,
                  'num_unrollings': args.num_unrollings,
                  'hidden_size': args.hidden_size,
                  'max_grad_norm': args.max_grad_norm,
                  'embedding_size': args.embedding_size,
                  'num_layers': args.num_layers,
                  'learning_rate': args.learning_rate}
        best_model = ''
    logging.info('parameters are:\n%s', params)
        
    # Read and split data.
    logging.info('Reading data from: %s', args.data_file)
    with codecs.open(args.data_file, 'r', encoding=args.encoding) as f:
        text = f.read()

    if args.test:
        text = text[:1000]
    logging.info('number of characters: %s', len(text))

    if args.debug:
        n = 10        
        logging.info('first %d characters: %s', n, text[:n])

    logging.info('Creating train, valid, test split')
    train_size = int(args.train_frac * len(text))
    valid_size = int(args.valid_frac * len(text))
    test_size = len(text) - train_size - valid_size
    train_text = text[:train_size]
    valid_text = text[train_size:train_size + valid_size]
    test_text = text[train_size + valid_size:]

    if args.vocab_file:
        vocab_index_dict, index_vocab_dict, vocab_size = load_vocab(args.vocab_file, args.encoding)
    else:
        logging.info('Creating vocabulary')
        vocab_index_dict, index_vocab_dict, vocab_size = create_vocab(text)
        vocab_file = os.path.join(args.output_dir, 'vocab.json')
        save_vocab(vocab_index_dict, vocab_file, args.encoding)
        logging.info('Vocabulary is saved in %s', vocab_file)
        args.vocab_file = vocab_file

    params['vocab_size'] = vocab_size
    logging.info('vocab size: %d', vocab_size)

    # Create batch generators.
    batch_size = params['batch_size']
    num_unrollings = params['num_unrollings']
    train_batches = BatchGenerator(train_text, batch_size, num_unrollings, vocab_size, 
                                   vocab_index_dict, index_vocab_dict)
    eval_train_batches = BatchGenerator(train_text, 1, 1, vocab_size,
                                        vocab_index_dict, index_vocab_dict)
    valid_batches = BatchGenerator(valid_text, 1, 1, vocab_size,
                                   vocab_index_dict, index_vocab_dict)
    test_batches = BatchGenerator(test_text, 1, 1, vocab_size,
                                  vocab_index_dict, index_vocab_dict)

    if args.debug:
        logging.info('Test batch generators')
        logging.info(batches2string(train_batches.next(), index_vocab_dict))
        logging.info(batches2string(valid_batches.next(), index_vocab_dict))
        logging.info('Show vocabulary')
        logging.info(vocab_index_dict)
        logging.info(index_vocab_dict)
        
    # Create graphs
    logging.info('Creating graph')
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('training'):
            train_model = CharRNN(is_training=True, **params)
        tf.get_variable_scope().reuse_variables()
        with tf.name_scope('evaluation'):
            valid_model = CharRNN(is_training=False, **params)
            saver = tf.train.Saver(name='checkpoint_saver')
            best_model_saver = tf.train.Saver(name='best_model_saver')

    if args.sample:
        if args.seed >= 0:
            np.random.seed(args.seed)
        # Sampling a sequence 
        with tf.Session(graph=graph) as session:
            saver.restore(session, args.init_model)
            sample = valid_model.sample_seq(session, args.length, args.start_text,
                                            vocab_index_dict, index_vocab_dict,
                                            max_prob=args.max_prob)
            print('\nstart text is:\n%s' % args.start_text)
            print('sampled text is:\n%s' % sample)
        return sample
    elif args.evaluate:
        example_batches = BatchGenerator(args.example_text, 1, 1, vocab_size,
                                         vocab_index_dict, index_vocab_dict)
        with tf.Session(graph=graph) as session:
            saver.restore(session, args.init_model)
            ppl = valid_model.run_epoch(session, len(args.example_text),
                                        example_batches,
                                        is_training=False)[0]
            print('\nexample text is: %s' % args.example_text)
            print('its perplexity is: %s' % ppl)
        return ppl            

    logging.info('Start training')
    logging.info('model size (number of parameters): %s', train_model.model_size)

    result = {}
    result['params'] = params
    result['vocab_file'] = args.vocab_file
    result['encoding'] = args.encoding

    try:
        # Use try and finally to make sure that intermediate
        # results are saved correctly so that training can
        # be continued later after interruption.
        with tf.Session(graph=graph) as session:
            # Version 8 changed the api of summary writer to use
            # graph instead of graph_def.
            if TF_VERSION >= 8:
                graph_info = session.graph
            else:
                graph_info = session.graph_def

            train_writer = tf.train.SummaryWriter(args.tb_log_dir + 'train/', graph_info)
            valid_writer = tf.train.SummaryWriter(args.tb_log_dir + 'valid/', graph_info)

            # load a saved model or start from random initialization.
            if args.init_model:
                saver.restore(session, args.init_model)
            else:
                tf.initialize_all_variables().run()
            for i in range(args.num_epochs):
                logging.info('\n')
                logging.info('Epoch %d\n', i)
                logging.info('Training on training set')
                # training step
                ppl, train_summary_str, global_step = train_model.run_epoch(
                    session,
                    train_size,
                    train_batches,
                    is_training=True,
                    verbose=args.verbose,
                    freq=args.progress_freq)
                # record the summary
                train_writer.add_summary(train_summary_str, global_step)
                train_writer.flush()
                # save model
                saved_path = saver.save(session, args.save_model,
                                                    global_step=train_model.global_step)
                logging.info('model saved in %s\n', saved_path)
                logging.info('Evaluate on validation set')
                valid_ppl, valid_summary_str, _ = valid_model.run_epoch(
                    session,
                    valid_size,
                    valid_batches, 
                    is_training=False,
                    verbose=args.verbose,
                    freq=args.progress_freq)
                # save and update best model
                if (not best_model) or (valid_ppl < best_valid_ppl):
                    best_model = best_model_saver.save(
                        session,
                        args.save_best_model,
                        global_step=train_model.global_step)
                    best_valid_ppl = valid_ppl
                valid_writer.add_summary(valid_summary_str, global_step)
                valid_writer.flush()
                logging.info('best model is saved in %s', best_model)
                logging.info('best validation ppl is %f\n', best_valid_ppl)
                result['latest_model'] = saved_path
                result['best_model'] = best_model
                # Convert to float because numpy.float is not json serializable.
                result['best_valid_ppl'] = float(best_valid_ppl)

            logging.info('latest model is saved in %s', saved_path)
            logging.info('best model is saved in %s', best_model)
            logging.info('best validation ppl is %f\n', best_valid_ppl)
            logging.info('Evaluate the best model on test set')
            saver.restore(session, best_model)
            test_ppl, _, _ = valid_model.run_epoch(session, test_size, test_batches,
                                                   is_training=False,
                                                   verbose=args.verbose,
                                                   freq=args.progress_freq)
            result['test_ppl'] = float(test_ppl)
    finally:
        result_path = os.path.join(args.output_dir, 'result.json')
        if os.path.exists(result_path):
            os.remove(result_path)
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, sort_keys=True)


def create_vocab(text):
    unique_chars = list(set(text))
    vocab_size = len(unique_chars)
    vocab_index_dict = {}
    index_vocab_dict = {}
    for i, char in enumerate(unique_chars):
        vocab_index_dict[char] = i
        index_vocab_dict[i] = char
    return vocab_index_dict, index_vocab_dict, vocab_size


def load_vocab(vocab_file, encoding):
    with codecs.open(vocab_file, 'r', encoding=encoding) as f:
        vocab_index_dict = json.load(f)
    index_vocab_dict = {}
    vocab_size = 0
    for char, index in vocab_index_dict.iteritems():
        index_vocab_dict[index] = char
        vocab_size += 1
    return vocab_index_dict, index_vocab_dict, vocab_size


def save_vocab(vocab_index_dict, vocab_file, encoding):
    with codecs.open(vocab_file, 'w', encoding=encoding) as f:
        json.dump(vocab_index_dict, f, indent=2, sort_keys=True)
        
if __name__ == '__main__':
    main()
