import tensorflow as tf
import numpy as np
import common.utils as utils
from lstm.train import construct_graph

num_steps = 100
max_length = 200

def update_feed():
  input = np.zeros([1, num_steps], dtype=np.int64)
  dictionary = utils.load_dictionary()
  input[b, 0] = dictionary['<bof>']
  return input

def sample(prediction):
  return np.random.choice(prediction.size, (), p=prediction)[()] # sample 1 word

def format_feed(input):
  """Format input to feed to LSTM
  Implicitly batch_size = 1
  """
  length = len(input)
  if length < num_steps:
    return [input + [0]*(num_steps-length)]
  else:
    return [input[-num_steps:]]

def generate():
  graph = tf.Graph()
  _, _, inputs, probs, _, dictionary, saver, _ = construct_graph(graph)
  with tf.Session(graph=graph) as sess:
    saver.restore(sess, "checkpoints/model.ckpt")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    feed_input = []
    sample_character = dictionary['<bof>']
    feed_input.append(sample_character) # begin with <bof>
    try:
      for i in range(max_length):
        feed_dict = {inputs: format_feed(feed_input)} # feed last <num_steps> words
        test_probs = probs.eval(feed_dict=feed_dict) # predict
        sample_character = sample(test_probs[0]) # sampling
        feed_input.append(sample_character) # update feed
        if sample_character == '<eof>': # stop sampling if meet <eof>
          break
    except Exception as e:
      print('Exception: {}'.format(e.args))
      coord.request_stop(e)

    utils.print_data(feed_input, pretty=True)
    
    coord.request_stop()
    coord.join(threads)  

if __name__ == '__main__':
  generate()