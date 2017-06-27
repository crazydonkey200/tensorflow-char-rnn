import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import argparse
import codecs
import json
import os

import numpy as np
from char_rnn_model import *
from train import load_vocab

def main():
    parser = argparse.ArgumentParser()
    
    # Parameters for using saved best models.
    parser.add_argument('--init_dir', type=str, default='',
                        help='continue from the outputs in the given directory')

    # Parameters for picking which model to use. 
    parser.add_argument('--model_path', type=str, default='',
                        help='path to the model file like output/best_model/model-40.')

    # Parameters for sampling.
    parser.add_argument('--temperature', type=float,
                        default=1.0,
                        help=('Temperature for sampling from softmax: '
                              'higher temperature, more random; '
                              'lower temperature, more greedy.'))

    parser.add_argument('--max_prob', dest='max_prob', action='store_true',
                        help='always pick the most probable next character in sampling')

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
    
    args = parser.parse_args()

    # Prepare parameters.
    with open(os.path.join(args.init_dir, 'result.json'), 'r') as f:
        result = json.load(f)
    params = result['params']

    if args.model_path:    
        best_model = args.model_path
    else:
        best_model = result['best_model']

    best_valid_ppl = result['best_valid_ppl']
    if 'encoding' in result:
        args.encoding = result['encoding']
    else:
        args.encoding = 'utf-8'
    args.vocab_file = os.path.join(args.init_dir, 'vocab.json')
    vocab_index_dict, index_vocab_dict, vocab_size = load_vocab(args.vocab_file, args.encoding)

    # Create graphs
    logging.info('Creating graph')
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('evaluation'):
            test_model = CharRNN(is_training=False, use_batch=False, **params)
            saver = tf.train.Saver(name='checkpoint_saver')

    if args.evaluate:
        example_batches = BatchGenerator(args.example_text, 1, 1, vocab_size,
                                         vocab_index_dict, index_vocab_dict)
        with tf.Session(graph=graph) as session:
            saver.restore(session, best_model)
            ppl = test_model.run_epoch(session, len(args.example_text),
                                        example_batches,
                                        is_training=False)[0]
            print('Example text is: %s' % args.example_text)
            print('Perplexity is: %s' % ppl)
    else:
        if args.seed >= 0:
            np.random.seed(args.seed)
        # Sampling a sequence 
        with tf.Session(graph=graph) as session:
            saver.restore(session, best_model)
            sample = test_model.sample_seq(session, args.length, args.start_text,
                                            vocab_index_dict, index_vocab_dict,
                                            temperature=args.temperature,
                                            max_prob=args.max_prob)
            print('Sampled text is:\n%s' % sample)
        return sample

if __name__ == '__main__':
    main()
