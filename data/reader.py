import tensorflow as tf
import common.utils as utils

def get_batch(batch_size, num_steps, name=None):
  """Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one."""
  data = utils.load_data()
  
  with tf.name_scope(name, "Input", [data, batch_size, num_steps]):
    data = tf.convert_to_tensor(data, name="data", dtype=tf.int32)

    data_len = tf.size(data)
    batch_len = data_len // batch_size
    data = tf.reshape(data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps])
    y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
    return x, y


