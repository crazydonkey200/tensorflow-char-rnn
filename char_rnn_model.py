import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import logging
import time
import numpy as np
import tensorflow as tf

# Disable Tensorflow logging messages.
logging.getLogger('tensorflow').setLevel(logging.WARNING)

class CharRNN(object):
  """Character RNN model."""
  
  def __init__(self, is_training, batch_size, num_unrollings, vocab_size, 
               hidden_size, max_grad_norm, embedding_size, num_layers,
               learning_rate, model, dropout=0.0, input_dropout=0.0, use_batch=True):
    self.batch_size = batch_size
    self.num_unrollings = num_unrollings
    if not use_batch:
      self.batch_size = 1
      self.num_unrollings = 1
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.max_grad_norm = max_grad_norm
    self.num_layers = num_layers
    self.embedding_size = embedding_size
    self.model = model
    self.dropout = dropout
    self.input_dropout = input_dropout
    if embedding_size <= 0:
      self.input_size = vocab_size
      # Don't do dropout on one hot representation.
      self.input_dropout = 0.0
    else:
      self.input_size = embedding_size
    self.model_size = (embedding_size * vocab_size + # embedding parameters
                       # lstm parameters
                       4 * hidden_size * (hidden_size + self.input_size + 1) +
                       # softmax parameters
                       vocab_size * (hidden_size + 1) +
                       # multilayer lstm parameters for extra layers.
                       (num_layers - 1) * 4 * hidden_size *
                       (hidden_size + hidden_size + 1))
    # self.decay_rate = decay_rate

    # Placeholder to feed in input and targets/labels data.
    self.input_data = tf.placeholder(tf.int64,
                                     [self.batch_size, self.num_unrollings],
                                     name='inputs')
    self.targets = tf.placeholder(tf.int64,
                                  [self.batch_size, self.num_unrollings],
                                  name='targets')

    if self.model == 'rnn':
      cell_fn = tf.contrib.rnn.BasicRNNCell
    elif self.model == 'lstm':
      cell_fn = tf.contrib.rnn.BasicLSTMCell
    elif self.model == 'gru':
      cell_fn = tf.contrib.rnn.GRUCell

    # params = {'input_size': self.input_size}
    params = {}
    if self.model == 'lstm':
      # add bias to forget gate in lstm.
      params['forget_bias'] = 0.0
      params['state_is_tuple'] = True
    # Create multilayer cell.
    cell = cell_fn(
        self.hidden_size, reuse=tf.get_variable_scope().reuse,
        **params)

    cells = [cell]
    # params['input_size'] = self.hidden_size
    # more explicit way to create cells for MultiRNNCell than
    # [higher_layer_cell] * (self.num_layers - 1)
    for i in range(self.num_layers-1):
      higher_layer_cell = cell_fn(
          self.hidden_size, reuse=tf.get_variable_scope().reuse,
          **params)
      cells.append(higher_layer_cell)

    if is_training and self.dropout > 0:
      cells = [tf.contrib.rnn.DropoutWrapper(
        cell,
        output_keep_prob=1.0-self.dropout)
               for cell in cells]

    multi_cell = tf.contrib.rnn.MultiRNNCell(cells)

    with tf.name_scope('initial_state'):
      # zero_state is used to compute the intial state for cell.
      self.zero_state = multi_cell.zero_state(self.batch_size, tf.float32)
      # Placeholder to feed in initial state.
      # self.initial_state = tf.placeholder(
      #   tf.float32,
      #   [self.batch_size, multi_cell.state_size],
      #   'initial_state')

      self.initial_state = create_tuple_placeholders_with_default(
        multi_cell.zero_state(batch_size, tf.float32),
        extra_dims=(None,),
        shape=multi_cell.state_size)      
      

    # Embeddings layers.
    with tf.name_scope('embedding_layer'):
      if embedding_size > 0:
        self.embedding = tf.get_variable(
          'embedding', [self.vocab_size, self.embedding_size])
      else:
        self.embedding = tf.constant(np.eye(self.vocab_size), dtype=tf.float32)

      inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)
      if is_training and self.input_dropout > 0:
        inputs = tf.nn.dropout(inputs, 1 - self.input_dropout)

    with tf.name_scope('slice_inputs'):
      # Slice inputs into a list of shape [batch_size, 1] data colums.
      sliced_inputs = [tf.squeeze(input_, [1])
                       for input_ in tf.split(axis=1, num_or_size_splits=self.num_unrollings, value=inputs)]
      
    # Copy cell to do unrolling and collect outputs.
    outputs, final_state = tf.contrib.rnn.static_rnn(
      multi_cell, sliced_inputs,
      initial_state=self.initial_state)

    self.final_state = final_state

    with tf.name_scope('flatten_ouputs'):
      # Flatten the outputs into one dimension.
      flat_outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size])

    with tf.name_scope('flatten_targets'):
      # Flatten the targets too.
      flat_targets = tf.reshape(tf.concat(axis=1, values=self.targets), [-1])
    
    # Create softmax parameters, weights and bias.
    with tf.variable_scope('softmax') as sm_vs:
      softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size])
      softmax_b = tf.get_variable("softmax_b", [vocab_size])
      self.logits = tf.matmul(flat_outputs, softmax_w) + softmax_b
      self.probs = tf.nn.softmax(self.logits)

    with tf.name_scope('loss'):
      # Compute mean cross entropy loss for each output.
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=self.logits, labels=flat_targets)
      self.mean_loss = tf.reduce_mean(loss)

    with tf.name_scope('loss_monitor'):
      # Count the number of elements and the sum of mean_loss
      # from each batch to compute the average loss.
      count = tf.Variable(1.0, name='count')
      sum_mean_loss = tf.Variable(1.0, name='sum_mean_loss')
      
      self.reset_loss_monitor = tf.group(sum_mean_loss.assign(0.0),
                                         count.assign(0.0),
                                         name='reset_loss_monitor')
      self.update_loss_monitor = tf.group(sum_mean_loss.assign(sum_mean_loss +
                                                               self.mean_loss),
                                          count.assign(count + 1),
                                          name='update_loss_monitor')
      with tf.control_dependencies([self.update_loss_monitor]):
        self.average_loss = sum_mean_loss / count
        self.ppl = tf.exp(self.average_loss)

      # Monitor the loss.
      loss_summary_name = "average loss"
      ppl_summary_name = "perplexity"
  
      average_loss_summary = tf.summary.scalar(loss_summary_name, self.average_loss)
      ppl_summary = tf.summary.scalar(ppl_summary_name, self.ppl)

    # Monitor the loss.
    self.summaries = tf.summary.merge([average_loss_summary, ppl_summary],
                                      name='loss_monitor')
    
    self.global_step = tf.get_variable('global_step', [],
                                       initializer=tf.constant_initializer(0.0))

    self.learning_rate = tf.constant(learning_rate)
    if is_training:
      # learning_rate = tf.train.exponential_decay(1.0, self.global_step,
      #                                            5000, 0.1, staircase=True)
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars),
                                        self.max_grad_norm)
      # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay_rate)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)

      self.train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                global_step=self.global_step)
      
  def run_epoch(self, session, data_size, batch_generator, is_training,
                verbose=0, freq=10, summary_writer=None, debug=False, divide_by_n=1):
    """Runs the model on the given data for one full pass."""
    # epoch_size = ((data_size // self.batch_size) - 1) // self.num_unrollings
    epoch_size = data_size // (self.batch_size * self.num_unrollings)
    if data_size % (self.batch_size * self.num_unrollings) != 0:
        epoch_size += 1

    if verbose > 0:
        logging.info('epoch_size: %d', epoch_size)
        logging.info('data_size: %d', data_size)
        logging.info('num_unrollings: %d', self.num_unrollings)
        logging.info('batch_size: %d', self.batch_size)

    if is_training:
      extra_op = self.train_op
    else:
      extra_op = tf.no_op()

    # Prepare initial state and reset the average loss
    # computation.
    state = session.run(self.zero_state)
    self.reset_loss_monitor.run()
    start_time = time.time()
    for step in range(epoch_size // divide_by_n):
      # Generate the batch and use [:-1] as inputs and [1:] as targets.
      data = batch_generator.next()
      inputs = np.array(data[:-1]).transpose()
      targets = np.array(data[1:]).transpose()

      ops = [self.average_loss, self.final_state, extra_op,
             self.summaries, self.global_step, self.learning_rate]

      feed_dict = {self.input_data: inputs, self.targets: targets,
                   self.initial_state: state}

      results = session.run(ops, feed_dict)
      average_loss, state, _, summary_str, global_step, lr = results
      
      ppl = np.exp(average_loss)
      if (verbose > 0) and ((step+1) % freq == 0):
        logging.info("%.1f%%, step:%d, perplexity: %.3f, speed: %.0f words",
                     (step + 1) * 1.0 / epoch_size * 100, step, ppl,
                     (step + 1) * self.batch_size * self.num_unrollings /
                     (time.time() - start_time))

    logging.info("Perplexity: %.3f, speed: %.0f words per sec",
                 ppl, (step + 1) * self.batch_size * self.num_unrollings /
                 (time.time() - start_time))
    return ppl, summary_str, global_step

  def sample_seq(self, session, length, start_text, vocab_index_dict,
                 index_vocab_dict, temperature=1.0, max_prob=True):

    state = session.run(self.zero_state)

    # use start_text to warm up the RNN.
    if start_text is not None and len(start_text) > 0:
      seq = list(start_text)
      for char in start_text[:-1]:
        x = np.array([[char2id(char, vocab_index_dict)]])
        state = session.run(self.final_state,
                            {self.input_data: x,
                             self.initial_state: state})
      x = np.array([[char2id(start_text[-1], vocab_index_dict)]])
    else:
      vocab_size = len(vocab_index_dict.keys())
      x = np.array([[np.random.randint(0, vocab_size)]])
      seq = []

    for i in range(length):
      state, logits = session.run([self.final_state,
                                   self.logits],
                                  {self.input_data: x,
                                   self.initial_state: state})
      unnormalized_probs = np.exp((logits - np.max(logits)) / temperature)
      probs = unnormalized_probs / np.sum(unnormalized_probs)

      if max_prob:
        sample = np.argmax(probs[0])
      else:
        sample = np.random.choice(self.vocab_size, 1, p=probs[0])[0]

      seq.append(id2char(sample, index_vocab_dict))
      x = np.array([[sample]])
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
    logging.info('Unexpected char %s', char)
    return 0


def id2char(index, index_vocab_dict):
  return index_vocab_dict[index]

    
def id2char_list(lst, index_vocab_dict):
  return [id2char(i, index_vocab_dict) for i in lst]


def create_tuple_placeholders_with_default(inputs, extra_dims, shape):
  if isinstance(shape, int):
    result = tf.placeholder_with_default(
      inputs, list(extra_dims) + [shape])
  else:
    subplaceholders = [create_tuple_placeholders_with_default(
      subinputs, extra_dims, subshape)
                       for subinputs, subshape in zip(inputs, shape)]
    t = type(shape)
    if t == tuple:
      result = t(subplaceholders)
    else:
      result = t(*subplaceholders)    
  return result

        
def create_tuple_placeholders(dtype, extra_dims, shape):
  if isinstance(shape, int):
    result = tf.placeholder(dtype, list(extra_dims) + [shape])
  else:
    subplaceholders = [create_tuple_placeholders(dtype, extra_dims, subshape)
                       for subshape in shape]
    t = type(shape)

    # Handles both tuple and LSTMStateTuple.
    if t == tuple:
      result = t(subplaceholders)
    else:
      result = t(*subplaceholders)
  return result
  
