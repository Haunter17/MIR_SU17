cleanFilePath = '(Taylor Swift) - 22.wav';
% find the file path without the extension to use in the names of the new files
[cleanPathstr, cleanName, cleanExt] = fileparts(cleanFilePath);

fprintf('==> Producing Additive White Gaussian Noise versions of signal in: %s\n', cleanFilePath)
% read in the clean signal
[cleanSignal, cleanFs] = audioread(cleanFilePath);



% Choose the variance of the gaussian (which
% sets the power of the noise) based on the SNR
% and the power of the original signal
SNRs = [-10, -5, 0, 5, 10, 15, 20,100];

Px = bandpower(cleanSignal); % calculate the average power per sample in the input
Pn = Px ./ (10.^(SNRs/10));% calculate the desired expected power per sample in the noise

% let N be a gaussian with mean meu and variance sigma^2
% N^2 will have an expected value of sigma^2
% Pn, the avg. power of the signal = E(N^2) = sigma^2,
% We know Pn, so we can find sigma = sqrt(Pn)
sigmas = sqrt(Pn);

% now we generate a new noise signal for each SNR
for i = 1:length(SNRs)
	% make the noise and add it onto the original signal
	fprintf('Making File for SNR = %f\n', SNRs(i))
	noiseSignal = normrnd(0, sigmas(i), length(cleanSignal), 1);
	fprintf('SNR ended up equal to: %f\n', snr(cleanSignal, noiseSignal))
	combinedSignal =  cleanSignal + noiseSignal;
	
	% write the combined signal to a .wav file
	newSignalFilename = sprintf('%s_AWGN_SNR=%f.wav', cleanName, SNRs(i));
	fprintf('Writing to %s', newSignalFilename)
	audiowrite(newSignalFilename, combinedSignal, cleanFs); % write it at the same sampling rate

end



% make a new .wav file for each SNR
% where the variance of the gaussian white noise
% will set the expected value of the power per sample
% this expected value for the power relative to the 
% average power per sample of the clean clean signal
% gives us our SNR. So SNR is a function of the input
% signal and the variance of the gaussian