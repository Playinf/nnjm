function file = skip_line(file, count)
  for j = 1:count
    if ~feof(file)
      fgetl(file);
    else
      break;
    end
  end
end
