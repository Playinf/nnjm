function [val, nn] = nnjm_compute(model, data)
  onum = model.output_size;
  [m, n] = size(data);

  if model.context + 1 ~= n
    error('ngram order does not match model');
  end

  x = data(:, 1:end - 1)';
  y = data(:, end)';
  y = sparse(y, 1:m, ones(1, m), onum, m);
  y = full(logical(y));
  input = model.embedding(:, x(:));
  input = reshape(input, model.context * model.feature_number, m);
  
  if isequal(model.device, 'gpu')
    input = gpuArray(input);
  end

  % traininig neural network
  net = model.network;
  net = ffnn_forward(net, input);
  
  z = sum(exp(net.output));
  logz = log(z);
  val = net.output(y);
  nn = net;
  
end
