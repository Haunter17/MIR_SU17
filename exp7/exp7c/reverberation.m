[y, fs] = audioread('../orig.wav');
reverb = reverberator('PreDelay',0.5, 'DecayFactor', 0.5, 'WetDryMix',1, 'SampleRate', fs);
yrev = step(reverb, y);
audiowrite('rev.wav', yrev, fs);
