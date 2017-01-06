function model = nnjm_init(model, options)
  model.options = options;
  % initialize parameter
  a = options.init_range(1);
  b = options.init_range(2);
  class = options.precision;
  % create fields for training
  model.network.error = cell(1, model.network.layer_number);
  model.network.gradient = cell(1, model.network.layer_number - 1);
  % create word embedding matrix on cpu
  feat_num = model.feature_number;
  word_num = model.input_size;
  model.embedding = a + (b - a) * rand(feat_num, word_num, class);

  handle = model.handle;

  % create synapse on computing device
  for i = 1:model.network.layer_number - 1
    m = model.layer_size(i);
    n = model.layer_size(i + 1);
    model.network.synapse{i} = a + (b - a) * handle.rand(m + 1, n, class);
    model.network.gradient{i} = handle.rand(m + 1, n, class);
  end
  
  if isequal(options.cost_function, 'selfnorm')
    n = model.network.layer_number;
    v = model.output_size;
    model.network.synapse{n - 1}(1, :) = -log(v);
  end
  
  if isequal(options.cost_function, 'logcost')
    model.network.cost_function = @logcost;
  else
    model.network.cost_function = @selfnorm;
  end
  
  model.epoch = 1;
  model.count = 0;
  model.cost = 0;

end