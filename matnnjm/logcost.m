function [cost, err] = logcost(nn, y, options)
  err = exp(nn.output);
  z = sum(err);
  logz = log(z);
  cost = -sum(nn.output(y)' - logz);
  z = repmat(z, size(err, 1), 1);
  err = err ./ z - y;
  cost = nn.handle.gather(cost);
end
