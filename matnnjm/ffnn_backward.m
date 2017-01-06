% ffnn_backward.m
% backward pass of neural network
function nn = ffnn_backward(nn, x)
  n = nn.layer_number;
  nn.error{n} = x .* derivative(nn.activation_function{n}, nn.neuron{n});
  
  for i = n - 1:-1:1
    act_func = nn.activation_function{i};
    nn.error{i} = nn.synapse{i}(2:end, :) * nn.error{i + 1};
    nn.error{i} = nn.error{i} .* derivative(act_func, nn.neuron{i});
  end
  
end
