% nnjm.m
% options.device: cpu or gpu
% options.precision: single or double
% options.parallel: max batch size
% options.window_size: window size
% options.input_size: input word size
% options.hidden_size: hidden layer size
% options.output_size: output layer size
% options.feature_number: feature_number
% options.init_range: initialization range
% options.activation_function: activation function of network
function model = nnltm(options)
  % load vocabulary
  model.input_vocab = read_vocab(options.input_vocabulary);
  model.output_vocab = read_vocab(options.output_vocabulary);

  model.device = options.device;
  model.precision = options.precision;
  model.window_size = options.window_size;
  model.input_size = size(model.input_vocab, 1);
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
  input_layer_size = model.window_size * feat_num;
  model.layer_size = [input_layer_size, model.hidden_size, model.output_size];
  options.size = model.layer_size;
  model.network = ffnn(options);

end
