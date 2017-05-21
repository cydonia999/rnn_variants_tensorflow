# -*- coding: utf-8 -*-
"""Module implementing variants of RNN Cells."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import LSTMStateTuple
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


class JZSCell(RNNCell):
  """JZS recurrent network cells.
  
  The implementation is based on:
  
        Rafal Jozefowicz and Wojciech Zaremba and Ilya Sutskever, 
        An Empirical Exploration of Recurrent Network Architectures,
        http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
  """

  def __init__(self, num_units, activation=tanh, architecture="JZS1"):
    """Initialize the JZS cell.

    Args:
      num_units: int, The number of units in the JZS cell.
      activation: Activation function of the inner states.
      architecture: Name of architecture among three variants(JZS1/JZS2/JZS3)
                    corresponding to MUT1/MUT2/MUT3 in the paper, respectively.
    """
    if architecture not in ['JZS1', 'JZS2', 'JZS3']:
      raise ValueError("%s: architecture must be one of JZS1/JZS2/JZS3.", self)
    self._num_units = num_units
    self._architecture = architecture
    self._activation = activation

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """JZS with num_units cells."""
    with vs.variable_scope(scope or "jzs_cell"):
      # We start with bias of 1.0 to not reset and not update.
      if self._architecture == 'JZS1':
        r, u = [inputs, state], [inputs]
      elif self._architecture == 'JZS2':
        r, u = [state], [inputs, state]
      elif self._architecture == 'JZS3':
        r, u = [inputs, state], [inputs, tanh(state)]
      with vs.variable_scope("gates_0"):
        r = _linear(r, self._num_units, True, 1.0, scope=scope)
        if self._architecture == 'JZS2':
          r = r + inputs
      with vs.variable_scope("gates_1"):
          u = _linear(u, self._num_units, True, 1.0, scope=scope)
      r, u = sigmoid(r), sigmoid(u)
      with vs.variable_scope("candidate"):
        if self._architecture == 'JZS1':
          c = _linear([r * state], self._num_units, True, scope=scope) + tanh(inputs)
        elif self._architecture in ['JZS2', 'JZS3']:
          c = _linear([inputs, r * state], self._num_units, True, scope=scope)
      c = self._activation(c)
      new_h = u * c + (1 - u) * state
    return new_h, new_h

class VariantLSTMCell(RNNCell):
  """Simplified Gating LSTM recurrent network cell.

  The implementation is based on:
        Yuzhen Lu, Fathi M. Salem,
        Simplified Gating in Long Short-term Memory (LSTM) Recurrent Neural Networks,  
        https://arxiv.org/abs/1701.03441

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.
  """

  def __init__(self, num_units, forget_bias=1.0, 
               activation=tanh, architecture='LS1'):
    """Initialize the Simplified Gating LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      activation: Activation function of the inner states.
      architecture: Name of architecture among three variants(LS1/LS2/LS3).
                    corresponding to LSTM1/LSTM2/LSTM3 in the paper, respectively.
    """
    if architecture not in ['LS1', 'LS2', 'LS3']:
      raise ValueError("%s: architecture must be one of LS1/LS2/LS3.", self)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = True # accepted and returned states are 2-tuples of the `c_state` and `m_state`.
    self._activation = activation
    self._architecture = architecture

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units) if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Simplified Gating LSTM."""
    with vs.variable_scope(scope or "simplified_gating_lstm_cell"):
      c, h = state
      with vs.variable_scope("gates_0") as gate_scope:
        if self._architecture in ['LS1', 'LS2']:
          concat = _linear([h], 3 * self._num_units, True if self._architecture == 'LS1' else False,
                  scope=scope)
          i, f, o = array_ops.split(value=concat, num_or_size_splits=3, axis=1)
        elif self._architecture == 'LS3':
          dtype = inputs.dtype
          bias = vs.get_variable( "bias", shape=[3 * self._num_units], dtype=dtype)
          i, f, o = array_ops.split(value=bias, num_or_size_splits=3, axis=0)

      with vs.variable_scope("gates_1"):
        j = _linear([inputs, h], self._num_units, True, scope=scope)


      new_c = c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j)
      new_h = self._activation(new_c) * sigmoid(o)

      new_state = LSTMStateTuple(new_c, new_h)
      return new_h, new_state


class SimplifiedLSTMCell(RNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.

  The default non-peephole implementation is based on:

    K. Greff, R. Srivastava, J. Koutnik, B. Steunebrink, J. Schmidhuber. LSTM: A Search Space Odyssey
    https://arxiv.org/abs/1503.04069

  """

  def __init__(self, num_units, 
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               forget_bias=1.0, activation=tanh, architecture='NOG'):
    """Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
        provided, then the projected values are clipped elementwise to within `[-proj_clip, proj_clip]`.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of the training.
      activation: Activation function of the inner states.
      architecture: Name of architecture among three variants(NOG/NFG/NIG/NIAF/NOAF/CIFG).
                NOG: No Output Gate.
                NFG: No Forget Gate.
                NIG: No Input Gate.
                NIAF: No Input Activation Function.
                NOAF: No Output Activation Function.
                CIFG: Coupled Input and Forget Gate.
                NP: No Peephole. Not implemented. Use tf.contrib.rnn.LSTMCell.
                FGR: Full Gate Recurrence. Not implemented.
    """
    if architecture not in ['NOG', 'NFG', 'NIG', 'NIAF', 'NOAF', 'CIFG']:
      raise ValueError("%s: architecture must be one of NOG, NFG, NIG, NIAF, NOAF, CIFG", self)

    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._forget_bias = forget_bias
    # state_is_tuple: False will be depricated. Forced to set True.
    self._state_is_tuple = True # accepted and returned states are 2-tuples of the `c_state` and `m_state`.
    self._activation = activation
    self._architecture = architecture

    if num_proj:
      self._state_size = ( LSTMStateTuple(num_units, num_proj) if self._state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = ( LSTMStateTuple(num_units, num_units) if self._state_is_tuple else 2 * num_units)
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def __call__(self, inputs, state, scope=None):
    """Run one step of simplified LSTM.

    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: This must be a tuple of state Tensors, both `2-D`, with column sizes `c_state` and `m_state`.
      scope: VariableScope for the created subgraph; defaults to "simplified_lstm_cell".

    Returns:
      A tuple containing:

      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via static shape inference.
    """
    num_proj = self._num_units if self._num_proj is None else self._num_proj

    (c_prev, m_prev) = state

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
    with vs.variable_scope(scope or "simplified_lstm_cell", initializer=self._initializer) as unit_scope:
      n_eqs = 3 if self._architecture in ['NOG', 'NFG', 'NIG', 'CIFG'] else 4 
      lstm_matrix = _linear([inputs, m_prev], n_eqs * self._num_units, bias=True, scope=scope)

      if self._architecture == 'NOG':
        i, j, f = array_ops.split( value=lstm_matrix, num_or_size_splits=3, axis=1)
      elif self._architecture in ['NFG', 'CIFG']:
        i, j, o = array_ops.split( value=lstm_matrix, num_or_size_splits=3, axis=1)
      elif self._architecture == 'NIG':
        j, f, o = array_ops.split( value=lstm_matrix, num_or_size_splits=3, axis=1)
      else:
        i, j, f, o = array_ops.split( value=lstm_matrix, num_or_size_splits=4, axis=1)

      # Diagonal connections
      if self._use_peepholes:
        with vs.variable_scope(unit_scope) as projection_scope:
          if self._num_unit_shards is not None:
            projection_scope.set_partitioner(None)
          if self._architecture not in ['NFG', 'CIFG']:
            w_f_diag = vs.get_variable( "w_f_diag", shape=[self._num_units], dtype=dtype)
          if self._architecture != 'NIG':
            w_i_diag = vs.get_variable( "w_i_diag", shape=[self._num_units], dtype=dtype)
          if self._architecture != 'NOG':
            w_o_diag = vs.get_variable( "w_o_diag", shape=[self._num_units], dtype=dtype)

      if self._use_peepholes:
        if self._architecture == 'NIG':
          c = sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev + self._activation(j)
        elif self._architecture == 'NFG':
          c = c_prev + sigmoid(i + w_i_diag * c_prev) * self._activation(j)
        elif self._architecture == 'NIAF':
          c = sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev + sigmoid(i + w_i_diag * c_prev) * j 
        elif self._architecture == 'CIFG':
          _i = sigmoid(i + w_i_diag * c_prev) 
          c = (1 - _i) * c_prev + _i * self._activation(j)
        else:
          c = sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev + sigmoid(i + w_i_diag * c_prev) * self._activation(j)
      else:
        if self._architecture == 'NIG':
          c = sigmoid(f + self._forget_bias) * c_prev + self._activation(j)
        elif self._architecture == 'NFG':
          c = c_prev + sigmoid(i) * self._activation(j)
        elif self._architecture == 'NIAF':
          c = sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) * j 
        elif self._architecture == 'CIFG':
          _i = sigmoid(i) 
          c = (1 - _i) * c_prev + _i * self._activation(j)
        else:
          c = sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) * self._activation(j)

      if self._cell_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
        # pylint: enable=invalid-unary-operand-type

      if self._use_peepholes:
        if self._architecture == 'NOG':
          m = self._activation(c)
        elif self._architecture == 'NOAF':
          m = sigmoid(o + w_o_diag * c) * c
        else:
          m = sigmoid(o + w_o_diag * c) * self._activation(c)
      else:
        if self._architecture == 'NOG':
          m = self._activation(c)
        elif self._architecture == 'NOAF':
          m = sigmoid(o) * c
        else:
          m = sigmoid(o) * self._activation(c)

      if self._num_proj is not None:
        with vs.variable_scope("projection") as proj_scope: 
          m = _linear(m, self._num_proj, bias=False, scope=scope)

        if self._proj_clip is not None:
          # pylint: disable=invalid-unary-operand-type
          m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
          # pylint: enable=invalid-unary-operand-type

    new_state = LSTMStateTuple(c, m)
    return m, new_state

