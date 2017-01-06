% softmax
% a safe version of softmax
function y = softmax(x)
  % max of each column
  [m, ~] = size(x);
  mval = max(x, [], 1);
  x = x - repmat(mval, m, 1);
  x = exp(x);
  z = repmat(sum(x, 1), m, 1);
  y = x ./ z;
end
