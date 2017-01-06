function y = rectified_linear(x)

    y = x;
    y(y < 0) = 0;

end