function save_nnltm(model)
  opts = model.options;
  file = fopen(opts.output, 'wb');
  
  % write options
  fprintf(file, '25\n');
  fprintf(file, 'model: nnltm\n');
  fprintf(file, 'order: %d\n', model.window_size + 1);
  fprintf(file, 'window: %d\n', model.window_size);
  fprintf(file, 'source-context: 0\n');
  fprintf(file, 'target-context: 0\n');
  fprintf(file, 'network: ffnn\n');
  fprintf(file, 'layer-number: %d\n', size(model.hidden_size, 1) + 2);
  fprintf(file, 'feature-number: %d\n', model.feature_number);
  fprintf(file, 'input-size: %d\n', model.input_size);
  fprintf(file, 'hidden-size: %s\n', num2str(model.hidden_size));
  fprintf(file, 'output-size: %d\n', model.output_size);
  fprintf(file, 'activation-function: %s\n', strjoin(opts.activation_function));
  fprintf(file, 'output-function: %s\n', opts.output_function);
  fprintf(file, 'learning-rate: %g\n', opts.learning_rate);
  fprintf(file, 'weight-decay: 0\n');
  fprintf(file, 'momentum: 0 0\n');
  fprintf(file, 'norm-control: %g\n', opts.control);
  fprintf(file, 'epoch: %d\n', opts.epoch);
  fprintf(file, 'max-epoch: %d\n', opts.max_epoch);
  fprintf(file, 'batch-size: %d\n', opts.batch_size);
  fprintf(file, 'init-method: none\n');
  fprintf(file, 'init-range: %s\n', num2str(opts.init_range, '%g'));
  fprintf(file, 'update-range: %s\n', num2str(opts.update_range, '%g'));
  fprintf(file, 'weight-range: %s\n', num2str(opts.weight_range, '%g'));
  fprintf(file, 'cost-function: %s\n', opts.cost_function);
  
  % write vocabulary
  write_vocab(file, {});
  write_vocab(file, model.input_vocab);
  write_vocab(file, model.output_vocab);
  
  % write weight
  [fnum, wnum] = size(model.embedding);
  fwrite(file, wnum, 'uint32');
  fwrite(file, fnum, 'uint32');
  fwrite(file, model.embedding, 'double');
  
  for i = 1:model.network.layer_number - 1
    [r, c] = size(model.network.synapse{i});
    fwrite(file, r, 'uint32');
    fwrite(file, c, 'uint32');
    synapse = model.handle.gather(model.network.synapse{i});
    fwrite(file, synapse, 'double');
  end
  
  fclose(file);
  
end