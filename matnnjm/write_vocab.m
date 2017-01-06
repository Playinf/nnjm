function write_vocab(file, vocab)
  n = size(vocab, 1);
  fwrite(file, n, 'uint32');
  
  for i = 1:n
    val = vocab{i};
    fprintf(file, '%s ||| %d\n', char(val{1}), val{2});
  end
end

