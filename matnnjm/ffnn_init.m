function nn = ffnn_init(nn, options)
  nn.error = cell(1, nn.layer_number);
  nn.gradient = cell(1, nn.layer_number - 1);
  handle = nn.handle;

  if isequal(options.init_method, 'random')
    for i = 1:nn.layer_number - 1
      m = nn.layer_size(i) + 1;
      n = nn.layer_size(i + 1);
      a = options.init_range(1);
      b = options.init_range(2);
      nn.synapse{i} = a + (b - a) * handle.rand(m, n);
    end
  end
  
  if isequal(options.cost_function, 'logcost')
    nn.cost_function = @logcost;
  else
    nn.cost_function = @selfnorm;
  end

end