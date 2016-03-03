import time
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn


class CharRNN(object):
  """Character RNN model."""
  
  def __init__(self, is_training, batch_size, num_unrollings, vocab_size, 
               hidden_size, max_grad_norm, embedding_size, num_layers):
    self.batch_size = batch_size
    self.num_unrollings = num_unrollings
    if not is_training:
      self.batch_size = 1
      self.num_unrollings = 1
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.max_grad_norm = max_grad_norm
    self.num_layers = num_layers
    self.embedding_size = embedding_size

    # Placeholder for input data
    self.input_data = tf.placeholder(tf.int64,
                                     [self.batch_size, self.num_unrollings])
    self.targets = tf.placeholder(tf.int64,
                                  [self.batch_size, self.num_unrollings])

    with tf.variable_scope('rnn_cell') as rnn_cell:
      # Create multilayer LSTM cell
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size,
                                               input_size=self.embedding_size,
                                               forget_bias=0.0)
      cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers)

    # Compute zeros_state to initialize the cell.
    self.zero_state = cell.zero_state(self.batch_size, tf.float32)

    # Placeholder for feeding initial state.
    self.initial_state = tf.placeholder(tf.float32,
                                        [self.batch_size, cell.state_size])

    # use embeddings
    with tf.device("/cpu:0"):
      self.embedding = tf.get_variable("embedding",
                                       [self.vocab_size, self.embedding_size])
      inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

    # Prepare the inputs.
    sliced_inputs = [tf.squeeze(input_, [1])
                     for input_ in tf.split(1, self.num_unrollings, inputs)]
    outputs, state = rnn.rnn(cell, sliced_inputs, initial_state=self.initial_state)

    flat_outputs = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
    with tf.variable_scope('softmax') as sm_vs:
      softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size])
      softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(flat_outputs, softmax_w) + softmax_b
    flat_targets = tf.reshape(tf.concat(1, self.targets), [-1])
    self.probs = tf.nn.softmax(logits)
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, flat_targets)
    
    self.mean_loss = tf.reduce_sum(loss) / (self.batch_size * self.num_unrollings)
    # Monitor the loss.
    loss_summary = tf.scalar_summary("cross entropy", self.mean_loss)
    # Merge all the summaries and write them out to /tmp/mnist_logs
    self.summaries = tf.merge_all_summaries()
    
    self.final_state = state
    
    
    if is_training:
      self.global_step = tf.Variable(0)
      learning_rate = tf.train.exponential_decay(1.0, self.global_step,
                                                 5000, 0.1, staircase=True)
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars),
                                        self.max_grad_norm)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
      self.saver = tf.train.Saver()
    
      
  def run_epoch(self, session, data_size, batch_generator, is_train,
                verbose=False, frequency=10, summary_writer=None, debug=False):
    """Runs the model on the given data for one full pass."""
    epoch_size = ((data_size // self.batch_size) - 1) // self.num_unrollings
    
    sum_loss = 0.0
    iters = 0
    state = self.zero_state.eval()
    if is_train:
      extra_op = self.train_op
    else:
      extra_op = tf.no_op()
    print('epoch_size: %d' % epoch_size)
    print('data_size: %d' % data_size)
    print('num_unrollings: %d' % self.num_unrollings)
    print('batch_size: %d' % self.batch_size)
    start_time = time.time()
    if debug:
      print('init state: %s' % state)
    for step in range(epoch_size):
      data = batch_generator.next()

      x = np.array(data[:-1]).transpose()
      y = np.array(data[1:]).transpose()
      if is_train:
        mean_loss, state, _, summary_str, global_step = session.run([self.mean_loss,
                                                                     self.final_state,
                                                                     extra_op,
                                                                     self.summaries,
                                                                     self.global_step],
                                                                    {self.input_data: x,
                                                                     self.targets: y,
                                                                     self.initial_state: state})
      else:
        mean_loss, state, _, probs  = session.run([self.mean_loss,
                                                   self.final_state,
                                                   extra_op,
                                                   self.probs],
                                                  {self.input_data: x,
                                                   self.targets: y,
                                                   self.initial_state: state})
        if debug:
          print('after step %d' % step)
          print('input: %s' % x)
          print('state: %s' % state)
          print('probs: %s' % probs[0])
          print('mean_loss: %s' % mean_loss)
          print('computed loss: %s' % -np.log(probs[0][y[0][0]]))

      sum_loss += mean_loss
      iters += 1 # self.num_unrollings
      ppl = np.exp(sum_loss / iters)
      if verbose and ((step+1) % frequency == 0):
        print("%.1f%%, step:%d, perplexity: %.3f, speed: %.0f wps" %
              (step * 1.0 / epoch_size * 100, step, ppl,
               iters * self.batch_size / (time.time() - start_time)))
      if is_train and (summary_writer is not None):
        summary_writer.add_summary(summary_str, global_step)

    print("final ppl: %.3f, speed: %.0f wps" %
          (ppl, iters * self.batch_size / (time.time() - start_time)))
    return ppl

  def sample_seq(self, session, length, vocab_index_dict, index_vocab_dict,
                 start_char=None, max_prob=True, prime_text=None):

    state = self.zero_state.eval()

    # use prime_text to warm up the RNN.
    if prime_text is not None:
      seq = list(prime_text)
      for char in prime_text[:-1]:
        x = np.array([[char2id(char, vocab_index_dict)]])
        state = session.run(self.final_state,
                            {self.input_data: x,
                             self.initial_state: state})
      x = np.array([[char2id(prime_text[-1], vocab_index_dict)]])
        
    elif start_char is not None:
      seq = [start_char]
      x = np.array([[char2id(start_char, vocab_index_dict)]])

    for i in range(length):
      
      # losses = []
      # next_states = []
      # print(state.shape)
      # print(x)
      # sum_loss = 0.0
      
      #for j in range(self.vocab_size):
      state, probs = session.run([self.final_state,
                                  self.probs],
                                 {self.input_data: x,
                                  self.initial_state: state})
                                             
      # next_states.append(next_state)
      # losses.append(loss)


      # if max_prob:
      #   # print(losses)
      #   print(x)
      #   sample = np.argmin(losses)
      #   print(losses)
      #   print(sample)
      #   sum_loss += losses[sample]
      #   mean_loss = (losses[sample])
      #   print('input: %s %s' % (x, index_vocab_dict[x[0][0]]))
      #   print('state: %s' % state)
      #   print('probs: %s' % probs)
      #   print('output: %s %s' % (sample, index_vocab_dict[sample]))
      #   print('mean_loss: %s' % loss)
      #   print('computed loss: %s' % -np.log(probs[0][sample]))
        
      # print(state.shape)
      # if not (np.sum(probs[0]) == 1.0):
      #   print(probs[0])
      #   # print(np.sum(probs[0]))
      # # assert np.sum(probs[0]) == 1.0
      if max_prob:
        sample = np.argmax(probs[0])
      else:
        sample = np.random.choice(self.vocab_size, 1, p=probs[0])[0]
      
      seq.append(id2char(sample, index_vocab_dict))
      x = np.array([[sample]])
      # state = next_states[sample]

    # print(np.exp(sum_loss / length))
    return ''.join(seq)
      
        
class BatchGenerator(object):
    """Generate and hold batches."""
    def __init__(self, text, batch_size, n_unrollings, vocab_size,
                 vocab_index_dict, index_vocab_dict):
      self._text = text
      self._text_size = len(text)
      self._batch_size = batch_size
      self.vocab_size = vocab_size
      self._n_unrollings = n_unrollings
      self.vocab_index_dict = vocab_index_dict
      self.index_vocab_dict = index_vocab_dict
      
      segment = self._text_size // batch_size
      # number of elements in cursor list is the same as
      # batch_size.  each batch is just the collection of
      # elements in where the cursors are pointing to.
      self._cursor = [ offset * segment for offset in range(batch_size)]
      self._last_batch = self._next_batch()
      
    def _next_batch(self):
      """Generate a single batch from the current cursor position in the data."""
      batch = np.zeros(shape=(self._batch_size), dtype=np.float)
      for b in range(self._batch_size):
        batch[b] = char2id(self._text[self._cursor[b]], self.vocab_index_dict)
        self._cursor[b] = (self._cursor[b] + 1) % self._text_size
      return batch

    def next(self):
      """Generate the next array of batches from the data. The array consists of
      the last batch of the previous array, followed by num_unrollings new ones.
      """
      batches = [self._last_batch]
      for step in range(self._n_unrollings):
        batches.append(self._next_batch())
      self._last_batch = batches[-1]
      return batches


# Utility functions
def batches2string(batches, index_vocab_dict):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, id2char_list(b, index_vocab_dict))]
  return s


def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)]


def char2id(char, vocab_index_dict):
  try:
    return vocab_index_dict[char]
  except KeyError:
    print('Unexpected char')
    return 0


def id2char(index, index_vocab_dict):
  return index_vocab_dict[index]

    
def id2char_list(lst, index_vocab_dict):
  return [id2char(i, index_vocab_dict) for i in lst]


def main(_):
  with open("tiny_shakespeare.txt", 'r') as f:
    text = f.read()

  print(text[100:200])
  print(len(text))
  text = text[:1000]

  # prepare data
  train_size = int(0.8 * len(text))
  valid_size = int(0.1 * len(text))
  test_size = len(text) - train_size - valid_size
  train_text = text[:train_size]
  valid_text = text[train_size:train_size + valid_size]
  test_text = text[train_size + valid_size:]

  print(train_size, train_text[:64])
  print(valid_size, valid_text[:64])
  print(test_size, test_text[:64])

  unique_chars = list(set(text))
  vocab_size = len(unique_chars)
  print('vocab size: %d' % vocab_size)
  vocab_index_dict = {}
  index_vocab_dict = {}

  for i, char in enumerate(unique_chars):
    vocab_index_dict[char] = i
    index_vocab_dict[i] = char

  batch_size = 1
  n_unrollings = 10
  train_batches = BatchGenerator(train_text, batch_size, n_unrollings, vocab_size)
  eval_train_batches = BatchGenerator(train_text, 1, 1, vocab_size)
  valid_batches = BatchGenerator(valid_text, 1, 1, vocab_size)
  test_batches = BatchGenerator(test_text, 1, 1, vocab_size)

  params = {'batch_size': batch_size, 'num_unrollings': n_unrollings,
            'vocab_size': vocab_size, 'hidden_size': 10,
            'max_grad_norm': 5.0, 'embedding_size': 10, 
            'num_layers': 1}
  
  tf.reset_default_graph()
  graph = tf.Graph()
  with graph.as_default():
    with tf.variable_scope('char_rnn') as scope:
      train_model = CharRNN(is_training=True, **params)
      tf.get_variable_scope().reuse_variables()
      valid_model = CharRNN(is_training=False, **params)
      test_model = CharRNN(is_training=False, **params)

  n_epochs = 10
  
  model_name = 'char_rnn'
  saved_model_dir = '/tmp/test_char_rnn'
  
  init_from_path = '' #'/tmp/saved_char_rnn_model-240'

  with tf.Session(graph=graph) as sess:
    if init_from_path:
      train_model.saver.restore(sess, init_from_path)
    else:
      tf.initialize_all_variables().run()
    for i in range(n_epochs):
      print('\nEpoch %d\n' % i)
      print('training')
      run_epoch(sess, train_model, train_size, train_batches,
                is_train=True, verbose=True)
      saved_path = train_model.saver.save(sess, saved_model_dir,
                                          global_step=train_model.global_step)
      print('model saved in %s\n' % saved_path)
      print('validation')
      run_epoch(sess, valid_model, valid_size, valid_batches,
                is_train=False, verbose=True)
    print('\ntest')
    run_epoch(sess, test_model, test_size, test_batches,
              is_train=False, verbose=True)


  train_model.saver.last_checkpoints
  with tf.Session(graph=graph) as sess:
    train_model.saver.restore(sess, train_model.saver.last_checkpoints[-1])
    for i in range(1):
      print('\n')
      print('='*80)
      print(sample_seq(sess, valid_model, 100, is_argmax=False,
                       prime_text='First Citizen:'))
      print('='*80)
      print('\n')


  try_text = 'salhjshakjbfsascasrs'
  try_text = train_text[20:40]
  try_batches = BatchGenerator(try_text, 1, 1, vocab_size=vocab_size)
  
  with tf.Session(graph=graph) as sess:
    train_model.saver.restore(sess, train_model.saver.last_checkpoints[-1])
    run_epoch(sess, valid_model, len(try_text), try_batches,
              is_train=False, verbose=True)

