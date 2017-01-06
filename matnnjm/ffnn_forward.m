% ffnn_forward
% compute output of neural network
% x: a m * n dimensional data, m is data size, n is batch size
function nn = ffnn_forward(nn, x)
  nn.neuron{1} = x;
  handle = nn.handle;
  
  for i = 2:nn.layer_number
    n = size(nn.neuron{i - 1}, 2);
    nn.neuron{i} = nn.synapse{i - 1}' * [handle.ones(1, n); nn.neuron{i - 1}];
    nn.neuron{i} = nn.activation_function{i}(nn.neuron{i});
  end
  
  nn.output = nn.output_function(nn.neuron{nn.layer_number});
  
end
