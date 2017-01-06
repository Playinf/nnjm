% nnjm_train.m
% all traininig are 1-based index, make sure to do preprocessing first
% data: m * n dimensional traininig ngram, m is data number, n is ngram size
% model: nnjm model
function [model, cost] = nnltm_train(model, data, opts)
  onum = model.output_size;
  [m, n] = size(data);

  if model.window_size + 1 ~= n
    error('training ngram order does not match model');
  end

  train = data(:, 1:end - 1)';
  label = data(:, end)';
  label = sparse(label, 1:m, ones(1, m), onum, m);
  label = full(logical(label));
  input = model.embedding(:, train(:));
  input = reshape(input, model.window_size * model.feature_number, m);

  if isequal(model.device, 'gpu')
    input = gpuArray(input);
  end

  % traininig neural network
  net = model.network;
  [net, cost] = ffnn_train(net, input, label, opts);

  % update parameter
  alpha = opts.learning_rate;
  min = opts.update_range(1);
  max = opts.update_range(2);
  num = net.layer_number;

  for i = 1:num - 1
    delta = constraint(alpha * net.gradient{i}, min, max, opts.device);
    %delta = alpha * net.gradient{i};
    net.synapse{i} = net.synapse{i} - delta;
    %net.synapse{i} = net.synapse{i} - alpha * net.gradient{i};
  end

  % update word vector
  index = train(:);
  errmat = reshape(net.error{1}, model.feature_number, m * model.window_size);
  delta = constraint(alpha * errmat, min, max, opts.device);

  if isequal(model.device, 'gpu')
    delta = gather(delta);
  end

  for i = 1:length(index)
    ind = index(i);
    model.embedding(:, ind) = model.embedding(:, ind) - delta(:, i);
  end

  model.network = net;

end
