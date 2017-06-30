function pitchShift(filename)
% filename: path to original audio file

addpath('phaseVocoder/');

% Shift key by +/- 1 whole tones for now.
pitchChange_list = [-1, 9, 8; -0.5, 106, 100; 0.5, 100, 106; 1, 8, 9];
[y, Fs]=audioread(filename); 
origlen = size(y,1);

for i = 1:4
    pitchChange = pitchChange_list(i,:);
    shiftY = pvoc(y, pitchChange(2)/pitchChange(3)); 
    newlen = size(shiftY,1);
    timeDilateY = resample(shiftY,origlen,newlen);
    audiowrite(strcat('audio/exp7/pitchShift_(',num2str(pitchChange(1)),').wav'),...
        timeDilateY, Fs);
end

end