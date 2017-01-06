function y = derivative(f, x)
  if isequal(f, @sigmoid)
    y = x .* (1 - x);
  elseif isequal(f, @tanh)
    y = 1 - x .* x;
  elseif isequal(f, @rectified_linear)
    y = ones(size(x), 'like', x);
    y(x <= 0) = 0;
  else
    y = ones(size(x), 'like', x);
  end
end
