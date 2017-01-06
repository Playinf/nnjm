function y = apply_contraint(x, a, b)
  y = max(a, x);
  y = min(b, y);
end