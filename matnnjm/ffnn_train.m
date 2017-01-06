% ffnn_train.m
% nn: neural network
% x: m * n dimensional training data, on cpu or gpu
% y: m * n dimensional 0-1 valued label, on cpu or gpu
% opts: additional options
function [nn, cost] = ffnn_train(nn, x, y, options)
  handle = nn.handle;

  % forward pass
  nn = ffnn_forward(nn, x);
  % calculate cost and error
  [cost, err] = nn.cost_function(nn, y, options);
  % backward pass
  nn = ffnn_backward(nn, err);
  
  % calculate gradient  
  for i = 1:nn.layer_number - 1
    n = size(nn.neuron{i}, 2);
    nn.gradient{i} = [handle.ones(1, n); nn.neuron{i}] * nn.error{i + 1}';
  end
  
  %cost = cost(1);
  
end
