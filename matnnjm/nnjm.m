% nnjm.m
% options.device: cpu or gpu
% options.precision: single or double
% options.parallel: max batch size
% options.source_context: source_context
% options.target_context: target_context
% options.input_size: input word size
% options.hidden_size: hidden layer size
% options.output_size: output layer size
% options.feature_number: feature_number
% options.init_range: initialization range
% options.activation_function: activation function of network
function model = nnjm(options)
  % load vocabulary
  model.source_vocab = read_vocab(options.source_vocabulary);
  model.target_vocab = read_vocab(options.target_vocabulary);
  model.output_vocab = read_vocab(options.output_vocabulary);

  model.device = options.device;
  model.precision = options.precision;
  model.source_context = options.source_context;
  model.target_context = options.target_context;
  model.input_size = size(model.source_vocab, 1) + size(model.target_vocab, 1);
  model.hidden_size = options.hidden_size;
  model.output_size = size(model.output_vocab, 1);
  model.feature_number = options.feature_number;

  % assign overloaded function
  if isequal(model.device, 'gpu')
    model.handle.ones = @gpuArray.ones;
    model.handle.rand = @gpuArray.rand;
    model.handle.zeros = @gpuArray.zeros;
    model.handle.gather = @gather;
  else
    model.handle.ones = @ones;
    model.handle.rand = @rand;
    model.handle.zeros = @zeros;
    model.handle.gather = @identity;
  end

  feat_num = model.feature_number;
  context = model.source_context + model.target_context;
  input_layer_size = context * feat_num;
  model.context = context;
  model.layer_size = [input_layer_size, model.hidden_size, model.output_size];
  options.size = model.layer_size;
  model.network = ffnn(options);

end
