WindowLen = 256;
AnalysisLen = 64;
SynthesisLen = 64;
saveFileName = '../same.wav';
Hopratio = SynthesisLen/AnalysisLen;

reader = dsp.AudioFileReader('../orig.wav', ...
  'SamplesPerFrame',AnalysisLen, ...
  'OutputDataType','double');
buff = dsp.Buffer(WindowLen, WindowLen - AnalysisLen);

% Create a Window System object, which is used for the ST-FFT. This object
% applies a window to the buffered input data.
win = dsp.Window('Hanning', 'Sampling', 'Periodic');
dft = dsp.FFT;

% Create an IFFT System object, which is used for the IST-FFT.
idft = dsp.IFFT('ConjugateSymmetricInput',true,'Normalize',false);
Fs = 22050;
player = audioDeviceWriter('SampleRate',Fs, ...
    'SupportVariableSizeInput',true, ...
    'BufferSize',512);

% Create a System object to log your data.
logger = dsp.SignalSink;
yprevwin = zeros(WindowLen-SynthesisLen,1);
gain = 1/(WindowLen*sum(hanning(WindowLen,'periodic').^2)/SynthesisLen);
unwrapdata = 2*pi*AnalysisLen*(0:WindowLen-1)'/WindowLen;
yangle = zeros(WindowLen,1);
firsttime = true;
while ~isDone(reader)
    y = reader();

%     player(y);    % Play back original audio

    % ST-FFT
    % FFT of a windowed buffered signal
    yfft = dft(win(buff(y)));

    % Convert complex FFT data to magnitude and phase.
    ymag       = abs(yfft);
    yprevangle = yangle;
    yangle     = angle(yfft);

    % Synthesis Phase Calculation
    % The synthesis phase is calculated by computing the phase increments
    % between successive frequency transforms, unwrapping them, and scaling
    % them by the ratio between the analysis and synthesis hop sizes.
    yunwrap = (yangle - yprevangle) - unwrapdata;
    yunwrap = yunwrap - round(yunwrap/(2*pi))*2*pi;
    yunwrap = (yunwrap + unwrapdata) * Hopratio;
    if firsttime
        ysangle = yangle;
        firsttime = false;
    else
        ysangle = ysangle + yunwrap;
    end

    % Convert magnitude and phase to complex numbers.
    ys = ymag .* complex(cos(ysangle), sin(ysangle));

    % IST-FFT
    ywin  = win(idft(ys));    % Windowed IFFT

    % Overlap-add operation
    olapadd  = [ywin(1:end-SynthesisLen,:) + yprevwin; ...
                ywin(end-SynthesisLen+1:end,:)];
    yistfft  = olapadd(1:SynthesisLen,:);
    yprevwin = olapadd(SynthesisLen+1:end,:);

    % Compensate for the scaling that was introduced by the overlap-add
    % operation
    yistfft = yistfft * gain;

    logger(yistfft);     % Log signal
end
release(reader);

loggedSpeech = logger.Buffer(1:end)';
player = audioDeviceWriter('SampleRate', Fs, ...
    'SupportVariableSizeInput', true, ...
    'BufferSize', 512);
% Play time-stretched signal
disp('Playing time-stretched signal...');
audiowrite(saveFileName, loggedSpeech.', Fs);
% player(loggedSpeech.');