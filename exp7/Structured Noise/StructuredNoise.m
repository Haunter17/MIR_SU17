% https://www.youtube.com/watch?v=tq-Bp_2FOGA - crowd cheering
% https://www.youtube.com/watch?v=8Q0Aqo6NV9s - restaurant ambience
% all files converted to mono channel and 22050 hz before being run through this
cleanFilePath = '(Taylor Swift) - 22.wav';
structuredNoisePath = 'CrowdSoundEffect.wav';
fprintf('==> Producing Additive Structured Noise versions of signal in: %s using noise from %s\n', cleanFilePath, structuredNoisePath)


% find the file path without the extension to use in the names of the new files
[cleanPathstr, cleanName, cleanExt] = fileparts(cleanFilePath);
[noisePathStr, noiseName, noiseExt] = fileparts(structuredNoisePath);
% read in the clean signal
[cleanSignal, cleanFs] = audioread(cleanFilePath);

% get the structured noise from a file - assume same sampling rate as clean signal
[structuredNoise, structuredFs] = audioread(structuredNoisePath);
% only keep the beginning of the noise, to match the length of the clean song
% assumes the clean song is shorter
structuredNoise = structuredNoise(1:length(cleanSignal));

% generate new files with the given signal to noise ratios
SNRs = [-15,-10, -5, 0, 5, 10, 15, 100];

Px = bandpower(cleanSignal); % calculate the average power per sample in the input
PstructuredNoise = bandpower(structuredNoise);
PnDesired = Px ./ (10.^(SNRs/10)); % calculate the desired average power per sample in the noise

% find how much to scale the structured noise by to get the desired average power
noiseScaling = sqrt(PnDesired / PstructuredNoise);

% scale the noise up for each SNR, combine it with the original signal, and write to a file
for i = 1:length(SNRs)
	fprintf('Making File for SNR = %f\n', SNRs(i));
	scaledNoise = structuredNoise * noiseScaling(i);
	fprintf('SNR ended up equal to: %f\n', snr(cleanSignal, scaledNoise));
	combinedSignal = cleanSignal + scaledNoise;
	% write the combined signal to a .wav file
	newSignalFilename = sprintf('%s_StructuredNoise_%s_SNR=%g.wav', cleanName, noiseName, SNRs(i));
	fprintf('Writing to %s', newSignalFilename);
	audiowrite(newSignalFilename, combinedSignal, cleanFs); % write it at the same sampling rate
end
