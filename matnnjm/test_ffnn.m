%function ffnn_test()
  % define options
  options.device = 'cpu';
  options.precision = 'double';
  options.size = [400 25 10];
  options.output_function = 'identity';
  options.activation_function = cell(1, 3);
  options.activation_function{1} = 'identity';
  options.activation_function{2} = 'rectified_linear';
  options.activation_function{3} = 'identity';
  options.cost_function = 'selfnorm';
  options.init_range = [-0.05, 0.05];
  options.init_method = 'random';
  options.alpha = 0.1;

  % create neural network
  nn = ffnn(options);
  nn = ffnn_init(nn, options);
  
  % prepare data
  batch = 1;
  %load('data.mat');
  %load('activation.mat');
  %load('ex4weights.mat');
  %nn.synapse{1} = Theta1';
  %nn.synapse{2} = Theta2';
  %data = data(1:400)';
  data = rand(options.size(1), batch);
  label = randi(options.size(3), batch, 1);
  y = full(sparse(label, 1:batch, ones(1, batch), options.size(3), batch));
  y = y == 1;
  
  % get numerical graident
  gold = cell(1, 2);
  gold{1} = zeros(options.size(1) + 1, options.size(2));
  gold{2} = zeros(options.size(2) + 1, options.size(3));
  epsilon = 1e-8;
  
  % get graident
  nn = ffnn_train(nn, data, y, options);
  gradient = nn.gradient;
  error = nn.error{1};
  delta = 0;
  
  for i = 1:options.size(1) + 1
    for j = 1:options.size(2)
      weight = nn.synapse{1}(i, j);
      nn.synapse{1}(i, j) = weight + epsilon;
      [nn, c1] = ffnn_train(nn, data, y, options);
      nn.synapse{1}(i, j) = weight - epsilon;
      [nn, c2] = ffnn_train(nn, data, y, options);
      gold{1}(i, j) = (c1 - c2) / (2 * epsilon);
      fprintf('%d,%d: %f|%f\n', i, j, gradient{1}(i, j), gold{1}(i, j));
      delta = delta + abs(gradient{1}(i, j) - gold{1}(i, j));
    end
  end
  
  for i = 1:options.size(2) + 1
    for j = 1:options.size(3)
      weight = nn.synapse{2}(i, j);
      nn.synapse{2}(i, j) = weight + epsilon;
      [nn, c1] = ffnn_train(nn, data, y, options);
      nn.synapse{2}(i, j) = weight - epsilon;
      [nn, c2] = ffnn_train(nn, data, y, options);
      gold{2}(i, j) = (c1 - c2) / (2 * epsilon);
      fprintf(1, '%d,%d: %f|%f\n', i, j, gradient{2}(i, j), gold{2}(i, j));
      delta = delta + abs(gradient{2}(i, j) - gold{2}(i, j));
    end
  end
  
  err = zeros(size(error));
  
  for i = 1:options.size(1)
    val = data(i);
    data(i) = val + epsilon;
    [nn, c1] = ffnn_train(nn, data, y, options);
    data(i) = val - epsilon;
    [nn, c2] = ffnn_train(nn, data, y, options);
    data(i) = val;
    err(i) = (c1 - c2) / (2 * epsilon);
    fprintf(1, '%d: %f|%f\n', i, error(i), err(i));
    delta = delta + abs(error(i) - err(i));
  end
  
  fprintf(1, 'delta: %f\n', delta);
  
%end