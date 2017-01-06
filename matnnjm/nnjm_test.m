function likelihood = nnjm_test(model, data)
  onum = model.output_size;
  [m, n] = size(data);

  if model.context + 1 ~= n
    error('ngram order does not match model');
  end

  x = data(:, 1:end - 1)';
  y = data(:, end)';
  y = full(sparse(y, 1:m, ones(1, m), onum, m));
  y = y == 1;
  input = model.embedding(:, x(:));
  input = reshape(input, model.context * model.feature_number, m);
  
  if isequal(model.device, 'gpu')
    input = gpuArray(input);
  end

  % computing
  net = model.network;
  net = ffnn_forward(net, input);
  output = softmax(net.output);
  likelihood = sum(log(output(y)));
  
end
