% test_nnltm.m
load('options-ltm.mat');
options.precision = 'single';
options.device = 'gpu';
% 
batch = 128;
options.batch = batch;
options.parallel = batch;
options.control = 0.1;
options.input_vocabulary = 'tcltm/t2s/input.vocab';
options.output_vocabulary = 'tcltm/t2s/output.vocab';
model = nnltm(options);
model = nnltm_init(model, options);
model_basename = 'model-iter-';
log_file = fopen('log.txt', 'a');

epoch = 40;
order = 15;
start = model.epoch;

prev = -1000000;
for i = start:epoch
  % training data
  file = fopen('tcltm/t2s/t2s.numberized', 'r');
  data = zeros(batch, order);
  cost = 0;
  count = 0;
  model_name = [model_basename, num2str(i)];
  model.options.output = [model_name, '.nnltm'];
  
  if model.count ~= 0
    file = skip_line(file, model.count);
    count = model.count;
    cost = model.cost;
  end
  
  time_begin = clock;
  
  % train one epoch
  while ~feof(file)
    n = 0;
    
    % read a batch
    for j = 1:batch
      if ~feof(file)
        line = fgetl(file);
        c = textscan(line, '%d');
        data(j, :) = c{1};
        n = n + 1;
        count = count + 1;
      else
        break;
      end
    end
  
    if n ~= 0
      data = data(1:n, :);
      data = data + 1;
      t1 = clock;
      [model, c] = nnltm_train(model, data, options);
      t2 = clock;
      time = etime(t2, t1);
      cost = cost + c;
    end
    
    if mod(count, 1000) == 0
      fprintf(1, 'word/s: %f\n', n / time);
    end
    
    % auto save
    if mod(count, 1000000) == 0
      name = model.options.output;
      model.options.output = 'autosave.nnjm';
      model.count = count;
      model.cost = cost;
      model.epoch = i;
      save('model-autosave', 'model');
      save_nnltm(model);
      model.options.output = name;
    end
  end
  
  model.cost = 0;
  model.count = 0;
  model.epoch = i;
  time_end = clock;
  time = etime(time_end, time_begin);
  fprintf(1, 'iter: %d, cost: %f, word/s: %f\n', i, cost / count, count / time);
  % write to log file
  fprintf(log_file, 'iter: %d, total-cost: %f, ', i, cost);
  fprintf(log_file, 'cost: %f, word/s: %f\n', cost / count, count / time);
  
  save(model_name, 'model');
  save_nnltm(model);
  fclose(file);
  
  % validate
%    file = fopen('small-data/validate.numberized', 'r');
%    data = zeros(batch, order);
%    likelihood = 0;
%    count = 0;
%    
%    t1 = clock;
%    while ~feof(file)
%      n = 0;
%      
%      % read a batch
%      for j = 1:batch
%        if ~feof(file)
%          line = fgetl(file);
%          c = textscan(line, '%d');
%          data(j, :) = c{1};
%          n = n + 1;
%          count = count + 1;
%        else
%          break;
%        end
%      end
%   
%     if n ~= 0
%       data = data(1:n, :);
%       data = data + 1;
%       l = nnjm_test(model, data);
%       likelihood = likelihood + l;
%     end
%   end
%   
%   t2 = clock;
%   time = etime(t2, t1);
%   
%   fprintf(1, 'likelihood: %f\n', likelihood / count);
%   fclose(file);
%   
%   likelihood = likelihood / count;
%   
%   if likelihood < prev
%     options.learning_rate = options.learning_rate * 0.5;
%   end
%   
%   %fprintf(1, '%f, %f', prev, likelihood);
%   prev = likelihood;
  
end