function y = constraint(x, a, b, device)
  if isequal(device, 'cpu')
    y = x;
    y(x > b) = b;
    y(x < a) = a;
  else
    y = arrayfun(@(x, a, b)(min(max(x, a), b)), x, a, b);
  end
end

