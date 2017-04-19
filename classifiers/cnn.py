import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C,H,W=input_dim
    self.params['W1']=np.random.standard_normal([num_filters,input_dim[0],filter_size,filter_size])*weight_scale
    self.params['b1']=np.zeros(num_filters)
    self.params['W2']=np.random.standard_normal([num_filters*H*W/4,hidden_dim])*weight_scale
    self.params['b2']=np.zeros(hidden_dim)
    self.params['W3']=np.random.standard_normal([hidden_dim,num_classes])*weight_scale
    self.params['b3']=np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out,cache={},{}
    out[0]=X
    out[1],cache['conv_relu_pool']=conv_relu_pool_forward(out[0],W1,b1,conv_param,pool_param)
    out[2],cache['affine_relu']=affine_relu_forward(out[1],W2,b2)
    out[3],cache['affine']=affine_forward(out[2],W3,b3)
    scores=out[3]


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss,dscore=softmax_loss(scores,y)
    loss+=0.5*self.reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3))
    dx={}
    dx[2],grads['W3'],grads['b3']=affine_backward(dscore,cache['affine'])
    dx[1],grads['W2'],grads['b2']=affine_relu_backward(dx[2],cache['affine_relu'])
    dx[0],grads['W1'],grads['b1']=conv_relu_pool_backward(dx[1],cache['conv_relu_pool'])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads




class ConvNet(object):
  """
  A four-layer convolutional network with the following architecture:
  
  conv - relu - BN - conv - relu - pool - BN - affine - relu - BN - affine- softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), nums_filters=[32,64], filter_sizes=[3,3],
               hidden_dim=500, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C,H,W=input_dim
    self.params['W1']=np.random.standard_normal([nums_filters[0],input_dim[0],filter_sizes[0],filter_sizes[0]])*weight_scale
    self.params['b1']=np.zeros(nums_filters[0])
    self.params['W2']=np.random.standard_normal([nums_filters[1],nums_filters[0],filter_sizes[1],filter_sizes[1]])*weight_scale
    self.params['b2']=np.zeros(nums_filters[1])
    self.params['W3']=np.random.standard_normal([nums_filters[1]*H*W/4,hidden_dim])*weight_scale
    self.params['b3']=np.zeros(hidden_dim)
    self.params['W4']=np.random.standard_normal([hidden_dim,num_classes])*weight_scale
    self.params['b4']=np.zeros(num_classes)
    self.params['gamma1']=np.random.randn(nums_filters[0]);
    self.params['beta1']=np.random.randn(nums_filters[0]);
    self.params['gamma2']=np.random.randn(nums_filters[1]);
    self.params['beta2']=np.random.randn(nums_filters[1]);
    self.params['gamma3']=np.random.randn(hidden_dim);
    self.params['beta3']=np.random.randn(hidden_dim);


    self.bn_params = []
    self.bn_params = [{'mode': 'train'} for i in xrange(3)]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):

    mode = 'test' if y is None else 'train'
    for bn_param in self.bn_params:
        bn_param[mode] = mode
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    gamma1,beta1=self.params['gamma1'],self.params['beta1']
    gamma2,beta2=self.params['gamma2'],self.params['beta2']
    gamma3,beta3=self.params['gamma3'],self.params['beta3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out,cache={},{}
    out[0]=X
    out[1],cache['conv_relu_BN']=conv_relu_BN_forward(out[0],W1,b1,gamma1,beta1,conv_param,self.bn_params[0])
    out[2],cache['conv_relu_pool_BN']=conv_relu_pool_BN_forward(out[1],W2,b2,gamma2,beta2,conv_param,pool_param,self.bn_params[1])
    out[3],cache['affine_relu_BN']=affine_relu_BN_forward(out[2],W3,b3,gamma3,beta3,self.bn_params[2])
    out[4],cache['affine']=affine_forward(out[3],W4,b4)
    scores=out[4]


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss,dscore=softmax_loss(scores,y)
    loss+=0.5*self.reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3)+np.sum(W4*W4))
    dx={}
    dx[3],grads['W4'],grads['b4']=affine_backward(dscore,cache['affine'])
    dx[2],grads['W3'],grads['b3'],grads['gamma3'],grads['beta3']=affine_relu_BN_backward(dx[3],cache['affine_relu_BN'])
    dx[1],grads['W2'],grads['b2'],grads['gamma2'],grads['beta2']=conv_relu_pool_BN_backward(dx[2],cache['conv_relu_pool_BN'])
    dx[0],grads['W1'],grads['b1'],grads['gamma1'],grads['beta1']=conv_relu_BN_backward(dx[1],cache['conv_relu_BN'])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads


  
  
pass




