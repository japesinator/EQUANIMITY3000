import signal
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import math

class QuantumCircuit():
  def __init__(self, bits, batch=1, classical_value=0):
    self.bits = bits
    self.shape = [batch]+[2]*bits
    self.state = tf.one_hot(indices=[classical_value]*batch, depth=2**bits)
    self.state = tf.reshape(self.state, shape=self.shape)

  def transform1b(self, bit, transform, control=None):
    if not control:
      self.state = tf.tensordot(transform, self.state, [[1], [bit+1]])
      return self.state
    else
      ident_indices = [[0]*4, [0,0,1,1]]
      gate_indices = list([0,0]+x for x in [[0,0],[0,1],[1,0],[1,1]])
      control_transform = tf.SparseTensor(
        indices = ident_indices + gate_indices,
        values = [1]*2 + transform[0] + transform[1],
        shape = [2]*4)
      return self.transform2b(bit, control, control_transform)

  def transform2b(self, bit1, bit2, transform):
    self.state = tf.tensordot(transform, self.state, [[2,3], [bit1+1,bit2+1]])
    return self.state

  def transform3b(self, bit1, bit2, bit3, transform):
    self.state = tf.tensordot(transform, self.state, [[3,4,5], [bit1+1,bit2+1,bit3+1]])
    return self.state

  #frac=0 is the id, frac=1 is 1 X gate (NOT), frac=2 is id again.
  def X(self, bit, frac=1, control=None):
    #if frac is 0 or frac is 1: #potential for optimization here.
    part = (1 + tf.exp(1j * math.pi * frac))/2
    transform = [[part, 1-part], [1-part, part]]
    return transform1b(self, bit, transform, control)

  def phase_shift(self, bit, rad, control=None):
    #This could also be done via slice/concat, but the mult is probably faster?
    #Definitely simpler at least
    exp_fact = tf.exp(1j * rad)

    """slice_rule = [slice(0,-1)]*(bit+1) + [0] + [slice(0,-1)]*(self.bits-bit-1)
    unchanged_slice = self.state.__getitem__(slice_rule)
    slice_rule[bit] = 1
    changed_slice   = self.state.__getitem__(slice_rule) * exp_fact
    self.state = tf.concat([unchanged_slice, changed_slice], axis=bit+1)"""
    
    transform = [[1,0],[0,exp_fact]]
    return transform1b(self, bit, transform, control)

  #frac=0 is the id, frac=1 is 1 Z gate, frac=2 is id again.
  def Z(self, bit, frac=1, control=None):
    #if frac is 0 or frac is 1: #potential for optimization here.
    phase_shift(bit, math*pi * frac, control)

  #frac=0 is the id, frac=1 is 1 Y gate, frac=2 is id again.
  def Y(self, bit, frac=1, control=None):
    #if frac is 0 or frac is 1: #potential for optimization here.
    part = (1 + tf.exp(1j * math.pi * frac))/2
    transform = [[part, 1j*(1-part)], [-1j*(1-part), part]]
    return transform1b(self, bit, transform, control)

  #Hadamard gate
  def H(self, bit, control=None):
    if not self.hadamard_mat:
      self.hadamard_mat = tf.constant([[1,1],[1,-1]])/math.sqrt(2)
    return transform1b(self, bit, self.hadamard_mat, control)

  def swap(self, bit1, bit2):
    #This could also be done via slice/concat, but the mult is probably faster?
    #Definitely simpler at least
    if not self.swap_mat:
      #self.swap_mat = tf.constant(
      #  [[[[1,0],[0,0]],[[0,0],[1,0]]],[[[0,1],[0,0]],[[0,0],[0,1]]]])
      self.swap_mat = tf.SparseTensor(
        indices=[[0]*4,[0,1,1,0],[1,0,0,1],[1]*4],
        values=[1]*4, shape = [2]*4)
    return transform2b(self, bit1, bit2, self.swap_mat)

  def cnot(self, bit1, bit2):
    if not self.cnot_mat:
      self.cnot_mat = tf.SparseTensor(
        indices=[[0]*4,[0,0,1,1],[1,1,0,1],[1,1,1,0]],
        values=[1]*4, shape = [2]*4)
    return transform2b(self, bit1, bit2, self.cnot_mat)
