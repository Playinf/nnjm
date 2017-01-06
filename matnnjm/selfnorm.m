% selfnorm_cost
% self normalization cost function
% cost = -logP + alpha * logZ^2
% cost: a vector contains cost information
% err: inital error used for backpropagation
function [cost, err] = selfnorm(nn, y, options)
  alpha = options.control;
  err = exp(nn.output);
  z = sum(err);
  logz = log(z);
  cost = -sum(nn.output(y)' - logz - alpha * logz .* logz);
  z = repmat(z, size(nn.output, 1), 1);
  logz = repmat(logz, size(nn.output, 1), 1);
  err = (1 + 2 * alpha * logz) .* (err ./ z) - y;
  cost = nn.handle.gather(cost);
end
