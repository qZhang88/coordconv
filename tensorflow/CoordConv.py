'''
From An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution (http://arxiv.org/abs/1807.03247)
'''

from tensorflow.python.layers import base
import tensorflow as tf


class AddCoords(base.Layer):
  """Add coords to a tensor"""
  def __init__(self, rank, with_r=False):
    super(AddCoords, self).__init__()
    self.rank = rank
    self.with_r = with_r
  
  def call(self, inputs):
    """
    input_tensor: 
      rank == 2: (batch, x_dim, y_dim, c)
      rank == 1: (batch, x_dim, c)
    """
    input_shape = inputs.get_shape().aslist()
    batch_size = input_shape[0]
    if self.rank ==1:
      x_dim = input[1]
      xx_channel = tf.tile(tf.expand_dims(tf.range(x_dim), 0), [batch_size, 1])
      xx_channel = tf.expand_dims(xx_range, -1)
      xx_channel = tf.cast(xx_channel, 'float32') / (x_dim - 1)
      xx_channel = xx_channel * 2 - 1
      ret = tf.concat([inputs, xx_channel], axis=-1)
      return ret
    
    if self.rank == 2
      x_dim, y_dim = input_shape[1:2]
      xx_ones = tf.ones([batch_size, y_dim], dtype=tf.int32)
      xx_ones = tf.expand_dims(xx_ones, 1)
      xx_range = tf.tile(tf.expand_dims(tf.range(x_dim), 0), [batch_size, 1])
      xx_range = tf.expand_dims(xx_range, -1)
      xx_channel = tf.matmul(xx_range, xx_ones)
      xx_channel = tf.expand_dims(xx_channel, -1)
      
      yy_ones = tf.ones([batch_size, x_dim], dtype=tf.int32)
      yy_ones = tf.expand_dims(yy_ones, -1)
      yy_range = tf.tile(tf.expand_dims(tf.range(y_dim), 0), [batch_size, 1])
      yy_range = tf.expand_dims(yy_range, 1)
      yy_channel = tf.matmul(yy_ones, yy_range)
      yy_channel = tf.expand_dims(yy_channel, -1)
      
      xx_channel = tf.cast(xx_channel, 'float32') / (x_dim - 1)
      yy_channel = tf.cast(yy_channel, 'float32') / (y_dim - 1)
      xx_channel = xx_channel * 2 - 1
      yy_channel = yy_channel * 2 - 1
      
      ret = tf.concat([inputs, xx_channel, yy_channel], axis=-1)
      
      if self.with_r:
        rr = tf.sqrt(tf.square(xx_channel - 0.5) + tf.square(yy_channel - 0.5))
        ret = tf.concat([ret, rr], axis=-1)
          
      return ret
    

class CoordConv2D(base.Layer):
  """CoordConv layer as in the paper."""
  def __init__(self, with_r, *args, **kwargs):
    super(CoordConv, self).__init__()
    self.addcoords = AddCoords(rank=2, with_r=with_r)
    self.conv = tf.layers.Conv2D(*args, **kwargs)
        
   def call(self, input_tensor):
     ret = self.addcoords(input_tensor)
     ret = self.conv(ret)
     return ret


class CoordConv1D(base.Layer):
  """CoordConv layer as in the paper."""
  def __init__(self, with_r, *args, **kwargs):
    super(CoordConv, self).__init__()
    self.addcoords = AddCoords(rank=1)
    self.conv = tf.layers.Conv1D(*args, **kwargs)
        
   def call(self, inputs):
     ret = self.addcoords(inputs)
     ret = self.conv(ret)
     return ret
