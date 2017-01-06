% nnjm_train.m
% all traininig are 1-based index, make sure to do preprocessing first
% data: m * n dimensional traininig ngram, m is data number, n is ngram size
% model: nnjm model
% function [model, cost] = nnjm_train(model, data, opts)
%   onum = opts.output_size;
%   [m, n] = size(data);
%   
%   if model.context + 1 ~= n
%     error('training ngram order does not match model');
%   end
%   
%   train = data(:, 1:end - 1)';
%   label = data(:, end)';
%   label = full(sparse(label, 1:m, ones(1, m), onum, m));
%   label = label == 1;
%   input = model.embedding(:, train(:));
%   input = reshape(input', model.context * model.feature_number, m);
%   
%   %% traininig neural network
%   net = model.network;
%   num = model.network.layer_number;
%   
%   % forward pass
%   if isequal(model.device, 'gpu')
%     net.neuron{1} = gpuArray(input);
%   else
%     net.neuron{1} = input;
%   end
%   
%   for i = 2:num
%     bias = model.handle.ones(1, m);
%     net.neuron{i} = net.synapse{i - 1}' * [bias; net.neuron{i - 1}];
%     net.neuron{i} = net.act_func{i}(net.neuron{i});
%   end
%   
%   % calculate cost
%   alpha = opts.control;
%   net.error{num} = exp(net.neuron{num});
%   facvec = log(sum(net.error{num}));
%   factor = repmat(facvec, size(net.error{num}, 1), 1);
%   net.error{num} = (1 + 2 * alpha) * net.error{num} .* factor - label;
%   cost = sum(-(net.neuron{num}(label)' - facvec - alpha * (facvec .* facvec)));
%   
%   % backward pass
%   for i = num - 1:-1:1
%     net.error{i} = net.synapse{i}(2:end, :) * net.error{i + 1};
%     net.error{i} = net.error{i} .* derivative(net.act_func{i}, net.neuron{i});
%   end
%   
%   % calculate gradient
%   for i = 1:num - 1
%     bias = model.handle.ones(1, m);
%     net.gradient{i} = [bias; net.neuron{i}] * net.error{i + 1}';
%   end
%   
%   %% update parameter
%   alpha = opts.learning_rate;
%   min = opts.update_range(1);
%   max = opts.update_range(2);
%   
%   for i = 1:num - 1
%     delta = constraint(alpha * net.gradient{i}, min, max, opts.device);
%     net.synapse{i} = net.synapse{i} - delta;
%   end
%   
%   % update word vector
%   index = train(:);
%   errmat = reshape(net.error{1}, model.feature_number, m * model.context);
%   delta = constraint(alpha * errmat, min, max, opts.device);
%   
%   if isequal(model.device, 'gpu')
%     delta = gather(delta);
%   end
%   
%   for i = 1:length(index)
%     ind = index(i);
%     model.embedding(:, ind) = model.embedding(:, ind) - delta(:, i);
%   end
%   
%   model.network = net;
%   
% end


% nnjm_train.m
% all traininig are 1-based index, make sure to do preprocessing first
% data: m * n dimensional traininig ngram, m is data number, n is ngram size
% model: nnjm model
function [model, cost] = nnjm_train(model, data, opts)
  onum = model.output_size;
  [m, n] = size(data);

  if model.context + 1 ~= n
    error('training ngram order does not match model');
  end

  train = data(:, 1:end - 1)';
  label = data(:, end)';
  label = sparse(label, 1:m, ones(1, m), onum, m);
  label = full(logical(label));
  input = model.embedding(:, train(:));
  input = reshape(input, model.context * model.feature_number, m);

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
  errmat = reshape(net.error{1}, model.feature_number, m * model.context);
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
