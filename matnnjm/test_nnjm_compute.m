 batch = 128;
 order = 15;
 file = fopen('small-data/validate.numberized', 'r');
 data = zeros(batch, order);
 count = 0;
 outfile = fopen('output.txt', 'w');
   
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
    val = nnjm_compute(model, data);
    fprintf(outfile, '%g\n', val);
  end
 end
  
fclose(file);
fclose(outfile)