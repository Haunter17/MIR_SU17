function ampSat (filename)

% Experiment 7e
scaleFactor_list = [-15, -10, -5, 0, 5, 10, 15];

for i = 1:7
    scaleFactor = scaleFactor_list(i);
    [y, Fs] = audioread(filename);
    y = y.*(10^(scaleFactor/10));
    audiowrite(strcat('audio/exp7/ampSat_',int2str(scaleFactor),'.wav'), y, Fs);
end

end