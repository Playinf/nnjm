function vocab = read_vocab(fname)
  file = fopen(fname, 'r');
  n = 0;
  
  while ~feof(file)
    fgetl(file);
    n = n + 1;
  end
  
  % rewind
  fseek(file, 0, -1);
  vocab = cell(n, 1);
  
  for i = 1:n
    line = fgetl(file);
    result = textscan(line, '%s ||| %d');
    vocab{i} = result;
  end
  
  fclose(file);
  
end