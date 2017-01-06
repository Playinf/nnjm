% ffnn.m
% ffnn, a feed-forward neural network
% device: computing device, cpu or gpu
% precision: computing precision, single or double
% layer_size: dimension of each layer
% layer_number: number of layer
% neuron: neuron of each layer
% synapse: synapse of each layer
% activation_function: activation function of each layer
% output_function: output function of each layer
function nn = ffnn(options)
  dim = options.size(:)';
  nn.device = options.device;
  nn.precision = options.precision;
  nn.layer_size = dim;
  nn.layer_number = length(dim);
  nn.neuron = cell(1, nn.layer_number);
  nn.synapse = cell(1, nn.layer_number - 1);
  nn.activation_function = cell(1, nn.layer_number);
  nn.output_function = cell(1, 1);
  
  % assign overloaded function
  if isequal(options.device, 'gpu')
    nn.handle.ones = @gpuArray.ones;
    nn.handle.rand = @gpuArray.rand;
    nn.handle.zeros = @gpuArray.zeros;
    nn.handle.gather = @gather;
  else
    nn.handle.ones = @ones;
    nn.handle.rand = @rand;
    nn.handle.zeros = @zeros;
    nn.handle.gather = @identity;
  end
    
  % assign activation function
  for i = 1:nn.layer_number
    if isequal(options.activation_function{i}, 'identity')
      nn.activation_function{i} = @identity;
    end
    
    if isequal(options.activation_function{i}, 'tanh')
      nn.activation_function{i} = @tanh;
    end
    
    if isequal(options.activation_function{i}, 'sigmoid')
      nn.activation_function{i} = @sigmoid;
    end
    
    if isequal(options.activation_function{i}, 'rectified_linear')
      nn.activation_function{i} = @rectified_linear;
    end
  end
  
  % assign output function
  if isequal(options.output_function, 'softmax')
    nn.output_function = @softmax;
  else
    nn.output_function = @identity;
  end
  
end
