% validate_nnjm.m
% validate
 batch = 128;
 order = 15;
 file = fopen('data/validate.numberized', 'r');
 data = zeros(batch, order);
 likelihood = 0;
 count = 0;
   
 t1 = clock;
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
    l = nnjm_test(model, data);
    likelihood = likelihood + l;
  end
end
  
t2 = clock;
time = etime(t2, t1);
  
fprintf(1, 'likelihood: %f\n', likelihood / count);
fclose(file);