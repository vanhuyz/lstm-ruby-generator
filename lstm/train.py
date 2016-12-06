import tensorflow as tf
import datetime
import os
import logging
import signal
import sys

import data.reader as reader
import common.utils as utils


logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()
logger.setLevel('INFO')
signal.signal(signal.SIGTERM, lambda s,f: sys.exit(0))

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'data', """Directory of data.""")
tf.app.flags.DEFINE_integer('batch_size', 32, """Batch size.""")
tf.app.flags.DEFINE_float('learning_rate', 1.0, """Learning rate.""")
tf.app.flags.DEFINE_integer('lstm_size', 128, """LSTM hidden size.""")
tf.app.flags.DEFINE_integer('num_layers', 2, """Number of LSTM layers.""")
tf.app.flags.DEFINE_integer('num_steps', 100, """Sequence length.""")

# Construct graph
def construct_graph(graph):
  size = FLAGS.lstm_size
  batch_size = FLAGS.batch_size
  num_steps = FLAGS.num_steps
  num_layers = FLAGS.num_layers

  dictionary = utils.load_dictionary()
  vocab_size = len(dictionary)

  with graph.as_default():
    inputs, labels = reader.get_batch(batch_size, num_steps)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)
    
    outputs = []
    with tf.variable_scope("RNN"):
      embedding = tf.get_variable("embedding", shape=[vocab_size, size], dtype=tf.float32)
      embed_inputs = tf.nn.embedding_lookup(embedding, inputs)
      outputs, state = tf.nn.dynamic_rnn(cell, embed_inputs, dtype=tf.float32)
      
    with tf.name_scope('loss'):
      output = tf.reshape(tf.concat(1, outputs), [-1, size])
      softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=tf.float32)
      softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
      logits = tf.matmul(output, softmax_w) + softmax_b
      targets = tf.reshape(labels, [-1])
      weights = tf.ones_like(targets, dtype=tf.float32)
      sequence_loss = tf.nn.seq2seq.sequence_loss_by_example(
          [logits],
          [targets],
          [weights])
      loss = tf.reduce_sum(sequence_loss) / batch_size # perplexity
      tf.scalar_summary('loss', loss)
      probs = tf.reshape(tf.nn.softmax(logits), tf.shape(logits))
    # Optimizer.
    with tf.name_scope('optimizer'):
      global_step = tf.Variable(0, trainable=False)
      learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate, global_step, 10000, 0.96, staircase=True)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      gradients, v = zip(*optimizer.compute_gradients(loss))
      gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
      optimizer = optimizer.apply_gradients(
        zip(gradients, v), global_step=global_step)
      tf.scalar_summary('learning_rate', learning_rate)
    
    # Saver
    saver = tf.train.Saver()

    # Summary
    summary = tf.merge_all_summaries()
  return optimizer, loss, inputs, probs, learning_rate, dictionary, saver, summary

def train():
  graph = tf.Graph()
  optimizer, loss, inputs, probs, learning_rate, dictionary, saver, summary = construct_graph(graph)

  today = datetime.date.today().strftime("%Y%m%d")

  train_writer = tf.train.SummaryWriter('tensorboard/train/{}'.format(today), graph)
  checkpoint_dir = "checkpoints/{}".format(today)

  with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
      step = 0
      while not coord.should_stop():
        _, train_loss, train_inputs, train_probs, train_lr, train_summary = sess.run([optimizer, loss, inputs, probs, learning_rate, summary])
        train_writer.add_summary(train_summary, step)
        train_writer.flush()
        
        if step % 1000 == 0:
          print("=" * 80)
          print("Loss at step {}: {}".format(step, train_loss))
          print("Train input:")
          utils.print_data(train_inputs[0])
          print("-" * 80)
          print("Train output:")
          utils.print_data(train_probs[0:FLAGS.num_steps].argmax(axis=1))
        if step % 10000 == 0:
          print("Learning rate: {}".format(train_lr))
          os.makedirs(checkpoint_dir, exist_ok=True)
          save_path = saver.save(sess, "checkpoints/{}/model.ckpt".format(today))
          print("Model saved in file: %s" % save_path)
        step += 1

    except KeyboardInterrupt:
      logger.warn('Interrupted')
      save_path = saver.save(sess, "checkpoints/{}/model.ckpt".format(today))
      logger.info("Model saved in file: %s" % save_path)      
      coord.request_stop()
    except SystemExit as e:
      logger.warn('Exited')
      save_path = saver.save(sess, "checkpoints/{}/model.ckpt".format(today))
      logger.info("Model saved in file: %s" % save_path)   
      coord.request_stop(e)
    except Exception as e:
      logger.error('Exception: {}'.format(e.args))
      coord.request_stop(e)
    finally:
      coord.request_stop()
      coord.join(threads)

if __name__ == '__main__':
  train()