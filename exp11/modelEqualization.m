clear;

% Number of corresponding ref files
refNum = [2, 11, 13, 14, 26, 36, 43, 51, 54, 65];

% List of path to ref files
filelist = 'audio/deathcabforcutie_shortref.list';

deltaPerSong = 10000;
duration = 14520; % Number of frames to average over, approx. 242 frames/s

fid = fopen(filelist);
curfile = fgetl(fid);

% Pre-allocate space to store delta reverb
deltaSample = zeros(121, deltaPerSong*length(refNum));

count = 1;

for i = 1:length(refNum)
    % Compute CQT coefficients for clean & corresponding noisy audio
    orig = computeQSpec(curfile);
    noise = computeQSpec(strcat('audio/createreverb/deathcabforcutie_fullval',int2str(i),'.wav'));

    logOrig = log(1+1000000*abs(orig.c));
    logNoise = log(1+1000000*abs(noise.c));

    % Shuffle the indices
    cleanIndex = randperm(size(logOrig,2)-14520);
    noiseIndex = randperm(size(logNoise,2)-14520);

    for j = 1:min(size(logNoise,2)-14520,deltaPerSong)
        % Get random starting index for 1-minute window
        c = cleanIndex(j);
        n = noiseIndex(j);
        
        aveOrig = mean(logOrig(:,c:c+duration),2);
        aveNoise = mean(logNoise(:,n:n+duration),2);
        
        deltaSample(:,count) = aveNoise - aveOrig;
        count = count + 1;
    end
end

deltaSample = deltaSample(:,1:count-1);
reverbSamples = deltaSample(:,randperm(size(deltaSample,2)));
reverbSamples = reverbSamples(:,1:100000);
save('reverbSamples_dcfc_newlog.mat', 'reverbSamples');
