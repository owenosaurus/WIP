function generate_wifi_lltf_dataset(snrDb, channelType)
% Generate Wi-Fi L-LTF dataset.
%
% Dataset format:
%   X = noisy LS channel estimate on 52 active L-LTF subcarriers
%   Y = true channel frequency response on 52 active L-LTF subcarriers
%
% For CBW20:
%   X: 52 complex -> IQ split -> 104 real columns
%   Y: 52 complex -> IQ split -> 104 real columns
%   One row: [X, Y] -> 208 columns


if nargin < 1 || isempty(snrDb)
    snrDb = 18;
end

if nargin < 2 || isempty(channelType)
    channelType = 'onetap';
end

channelType = lower(char(channelType));
assert(ismember(channelType, {'onetap','rayleigh','rician'}), ...
    'channelType must be ''onetap'', ''rayleigh'', or ''rician''.');

%% Dataset sizes
numMainSamples = 100000;
numEvalSamples = 20000;

%% Output files
% Kept compatible with the original filename convention.
baseName = sprintf('dataset_%ddb', round(snrDb));
mainFile = sprintf('%s.csv', baseName);
evalFile = sprintf('%s_eval.csv', baseName);

%% Basic parameters
cbw = 'CBW20';
fcHz = 2.412e9;

% Use FFT exactly after the L-LTF guard interval.
% This aligns the LS estimate with the physical CFR convention.
lltfSymOffset = 1.0;

%% Channel parameters
numTapsRange = [3 5];
maxDelaySamples = 8;
pdpTauSamples = 1.5;

% Rician K-factor parameters, in dB before conversion to linear scale.
kDbMean = 3.0;
kDbStd  = 2.0;
kDbMin  = 0.0;
kDbMax  = 7.0;

pSecondWeakRician = 0.10;
kDbMeanWeak = 1.5;
kDbStdWeak  = 1.0;
kDbMinWeak  = 0.0;
kDbMaxWeak  = 4.0;

% Mobility / Doppler
speedRangeMps = [0 5];

% Random start time for fading process
randomStartTimeMaxSec = 0.5;

%% Seeds
mainSeed = 94;
evalSeed = 3094;

%% Checks
assert(isscalar(snrDb) && isfinite(snrDb), ...
    'snrDb must be a finite scalar.');
assert(numTapsRange(1) >= 1, ...
    'numTapsRange(1) must be >= 1.');
assert(numTapsRange(1) <= numTapsRange(2), ...
    'Invalid numTapsRange.');
assert(speedRangeMps(1) <= speedRangeMps(2), ...
    'Invalid speedRangeMps.');

%% WLAN setup
cfg = wlanNonHTConfig('ChannelBandwidth', cbw);
Fs = wlanSampleRate(cfg);

cLight = 299792458;
lambda = cLight / fcHz;

txLLTF = wlanLLTF(cfg);
txLLTF = txLLTF(:);

ofdmInfo = wlanNonHTOFDMInfo('L-LTF', cbw);
nfft = ofdmInfo.FFTLength;
numTones = ofdmInfo.NumTones;

% Do not assert ofdmInfo.CPLength == 32.
% In some MATLAB versions, CPLength for L-LTF is not scalar.
% For CBW20 L-LTF:
%   total length = 160
%   useful symbols = 2 * 64
%   GI2 length = 160 - 128 = 32
lltfGI2Length = numel(txLLTF) - 2*nfft;

assert(isequal(numel(txLLTF), 160), ...
    'Expected 160 complex L-LTF time samples for CBW20.');
assert(isequal(nfft, 64), ...
    'Expected FFT length 64 for CBW20 L-LTF.');
assert(isequal(lltfGI2Length, 32), ...
    'Expected L-LTF GI2 length 32 for CBW20.');
assert(isequal(numTones, 52), ...
    'Expected 52 active tones for CBW20 L-LTF.');

lltfUsefulSampleIdx = lltfGI2Length + (1:(2*nfft));

% X = 52 complex noisy LS estimates -> 104 real IQ columns.
% Y = 52 complex true CFR values     -> 104 real IQ columns.
numInputCols = numTones * 2;
numLabelCols = numTones * 2;
numTotalCols = numInputCols + numLabelCols;

%% Shared config
P = struct();

P.channelType = channelType;

P.cbw = cbw;
P.fcHz = fcHz;
P.cfg = cfg;
P.Fs = Fs;
P.lambda = lambda;
P.txLLTF = txLLTF;

P.ofdmInfo = ofdmInfo;
P.nfft = nfft;
P.numTones = numTones;
P.lltfGI2Length = lltfGI2Length;
P.lltfUsefulSampleIdx = lltfUsefulSampleIdx;
P.lltfSymOffset = lltfSymOffset;

P.numInputCols = numInputCols;
P.numLabelCols = numLabelCols;
P.numTotalCols = numTotalCols;

P.numTapsRange = numTapsRange;
P.maxDelaySamples = maxDelaySamples;
P.pdpTauSamples = pdpTauSamples;

P.kDbMean = kDbMean;
P.kDbStd = kDbStd;
P.kDbMin = kDbMin;
P.kDbMax = kDbMax;

P.pSecondWeakRician = pSecondWeakRician;
P.kDbMeanWeak = kDbMeanWeak;
P.kDbStdWeak = kDbStdWeak;
P.kDbMinWeak = kDbMinWeak;
P.kDbMaxWeak = kDbMaxWeak;

P.speedRangeMps = speedRangeMps;
P.randomStartTimeMaxSec = randomStartTimeMaxSec;

% ofdmChannelResponse is available from R2023a.
% If unavailable, a manual CFR fallback is used.
P.useOfdmChannelResponse = (exist('ofdmChannelResponse', 'file') == 2);

%% Summary
fprintf('Configuration summary\n');
fprintf('  Channel type       : %s\n', channelType);
fprintf('  Bandwidth          : %s\n', cbw);
fprintf('  Sample rate        : %.3f MHz\n', Fs/1e6);
fprintf('  Center frequency   : %.6f GHz\n', fcHz/1e9);
fprintf('  SNR (dB)           : %.1f\n', snrDb);
fprintf('  FFT length         : %d\n', nfft);
fprintf('  L-LTF symOffset    : %.2f\n', lltfSymOffset);
fprintf('  Active tones       : %d\n', numTones);

if P.useOfdmChannelResponse
    fprintf('  True CFR source    : ofdmChannelResponse\n');
else
    fprintf('  True CFR source    : manual fallback from path gains/path filters\n');
end

%% Generate datasets
mainData = generateOneDataset(numMainSamples, snrDb, P, mainSeed, 'main');
writematrix(mainData, mainFile);

evalData = generateOneDataset(numEvalSamples, snrDb, P, evalSeed, 'eval');
writematrix(evalData, evalFile);

%% Done
fprintf('\nSaved datasets:\n');
fprintf('  %s  (%d x %d)\n', mainFile, size(mainData,1), size(mainData,2));
fprintf('  %s  (%d x %d)\n', evalFile, size(evalData,1), size(evalData,2));
fprintf('  X columns : 1 ~ %d\n', numInputCols);
fprintf('  Y columns : %d ~ %d\n', numInputCols+1, numInputCols+numLabelCols);

end


function DATA = generateOneDataset(numSamples, snrDb, P, baseSeed, tag)
rng(baseSeed);

DATA = zeros(numSamples, P.numTotalCols, 'single');

fprintf('Generating %s dataset (%d samples)...\n', tag, numSamples);
tic;

for n = 1:numSamples
    %% 1) Clean received L-LTF and true CFR label
    [rxLLTFClean, H52TrueCFR] = passThroughSelectedChannel(P, baseSeed, n);

    %% 2) Input X = noisy LS channel estimate on active L-LTF tones
    rxLLTFNoisy = addAwgnBySNR(rxLLTFClean, snrDb);
    H52NoisyLS = estimateLLTFLSChannel(rxLLTFNoisy, P.cfg, P.lltfSymOffset);

    assert(numel(H52NoisyLS) == P.numTones, ...
        'Unexpected number of complex LS estimates for X.');
    assert(numel(H52TrueCFR) == P.numTones, ...
        'Unexpected number of complex true CFR values for Y.');

    %% 3) Save row
    DATA(n,:) = single([ ...
        complexToIQIQRow(H52NoisyLS), ...
        complexToIQIQRow(H52TrueCFR)]);

    if mod(n, 5000) == 0 || n == numSamples
        fprintf('  %6d / %6d done\n', n, numSamples);
    end
end

elapsedSec = toc;
fprintf('Finished %s dataset in %.2f sec\n\n', tag, elapsedSec);

end


function H52LS = estimateLLTFLSChannel(rxLLTF, cfg, symOffset)
% Estimate LS channel on active L-LTF subcarriers.
%
% rxLLTF
%   -> wlanLLTFDemodulate: OFDM demodulation / FFT / active-tone extraction
%   -> wlanLLTFChannelEstimate: LS channel estimation using known L-LTF
%   -> H52LS: 52 complex values for CBW20

demodLLTF = wlanLLTFDemodulate(rxLLTF, cfg, symOffset);
chEst = wlanLLTFChannelEstimate(demodLLTF, cfg);

% SISO: chEst dimension = [numActiveSubcarriers, 1, 1]
H52LS = chEst(:,1,1);
H52LS = H52LS(:);

end


function [rxLLTF, H52TrueCFR] = passThroughSelectedChannel(P, baseSeed, sampleIdx)

switch P.channelType
    case 'onetap'
        seedNow = baseSeed + sampleIdx - 1;

        chan = createStdOneTapRayleighChannel(P.Fs, seedNow);

        [rxLLTF, H52TrueCFR] = runChannelAndGetTrueCFR( ...
            chan, P.txLLTF, 0, 0, P);

    case 'rayleigh'
        [pathDelaysSec, avgPathGainsDb, maxDelaySamp, ...
            thisMaxFdHz, seedNow, initTime] = ...
            sampleChannelProfile(P, baseSeed, sampleIdx);

        chan = createStdRayleighChannel(P.Fs, pathDelaysSec, ...
            avgPathGainsDb, thisMaxFdHz, seedNow);

        [rxLLTF, H52TrueCFR] = runChannelAndGetTrueCFR( ...
            chan, P.txLLTF, maxDelaySamp, initTime, P);

    case 'rician'
        [pathDelaysSec, avgPathGainsDb, maxDelaySamp, ...
            thisMaxFdHz, seedNow, initTime, numTaps] = ...
            sampleChannelProfile(P, baseSeed, sampleIdx);

        % KFactor must be linear scale, not dB.
        kVec = zeros(1, numTaps);

        kDb1 = sampleTruncGaussian(P.kDbMean, P.kDbStd, ...
            P.kDbMin, P.kDbMax);
        kVec(1) = 10^(kDb1 / 10);

        % Optional weak LOS on second path.
        if numTaps >= 2 && rand < P.pSecondWeakRician
            kDb2 = sampleTruncGaussian(P.kDbMeanWeak, P.kDbStdWeak, ...
                P.kDbMinWeak, P.kDbMaxWeak);
            kVec(2) = 10^(kDb2 / 10);
        end

        % LOS parameters only for paths with positive KFactor.
        losFd = zeros(1, numTaps);
        losPhase = zeros(1, numTaps);

        idxLos = (kVec > 0);
        losFd(idxLos) = ...
            (2 * rand(1, nnz(idxLos)) - 1) * thisMaxFdHz;
        losPhase(idxLos) = ...
            2 * pi * rand(1, nnz(idxLos));

        chan = createStdRicianChannel(P.Fs, pathDelaysSec, ...
            avgPathGainsDb, kVec, losFd, losPhase, ...
            thisMaxFdHz, seedNow);

        [rxLLTF, H52TrueCFR] = runChannelAndGetTrueCFR( ...
            chan, P.txLLTF, maxDelaySamp, initTime, P);

    otherwise
        error('Unsupported channelType: %s', P.channelType);
end

end


function chan = createStdOneTapRayleighChannel(Fs, seedNow)
% One-tap flat Rayleigh block fading.
% PathDelays = 0 and AveragePathGains = 0 define a flat channel.

chan = comm.RayleighChannel( ...
    'SampleRate', Fs, ...
    'PathDelays', 0, ...
    'AveragePathGains', 0, ...
    'NormalizePathGains', true, ...
    'MaximumDopplerShift', 0, ...
    'FadingTechnique', 'Sum of sinusoids', ...
    'InitialTimeSource', 'Input port', ...
    'RandomStream', 'mt19937ar with seed', ...
    'Seed', seedNow, ...
    'PathGainsOutputPort', true);

end


function chan = createStdRayleighChannel(Fs, pathDelaysSec, ...
    avgPathGainsDb, maxFdHz, seedNow)

chan = comm.RayleighChannel( ...
    'SampleRate', Fs, ...
    'PathDelays', pathDelaysSec, ...
    'AveragePathGains', avgPathGainsDb, ...
    'NormalizePathGains', true, ...
    'MaximumDopplerShift', maxFdHz, ...
    'FadingTechnique', 'Sum of sinusoids', ...
    'InitialTimeSource', 'Input port', ...
    'RandomStream', 'mt19937ar with seed', ...
    'Seed', seedNow, ...
    'PathGainsOutputPort', true);

end


function chan = createStdRicianChannel(Fs, pathDelaysSec, ...
    avgPathGainsDb, kVec, losFd, losPhase, maxFdHz, seedNow)

chan = comm.RicianChannel( ...
    'SampleRate', Fs, ...
    'PathDelays', pathDelaysSec, ...
    'AveragePathGains', avgPathGainsDb, ...
    'NormalizePathGains', true, ...
    'KFactor', kVec, ...
    'DirectPathDopplerShift', losFd, ...
    'DirectPathInitialPhase', losPhase, ...
    'MaximumDopplerShift', maxFdHz, ...
    'FadingTechnique', 'Sum of sinusoids', ...
    'InitialTimeSource', 'Input port', ...
    'RandomStream', 'mt19937ar with seed', ...
    'Seed', seedNow, ...
    'PathGainsOutputPort', true);

end


function [pathDelaysSec, avgPathGainsDb, maxDelaySamp, ...
    thisMaxFdHz, seedNow, initTime, numTaps] = ...
    sampleChannelProfile(P, baseSeed, sampleIdx)

numTaps = randi(P.numTapsRange);

delaysSamp = sampleDelayProfile(numTaps, P.maxDelaySamples);
pathDelaysSec = delaysSamp / P.Fs;

pdp = exp(-delaysSamp / P.pdpTauSamples);
avgPathGainsDb = 10 * log10(pdp);

thisSpeedMps = P.speedRangeMps(1) + ...
    (P.speedRangeMps(2) - P.speedRangeMps(1)) * rand;

thisMaxFdHz = thisSpeedMps / P.lambda;

seedNow = baseSeed + sampleIdx - 1;
initTime = P.randomStartTimeMaxSec * rand;
maxDelaySamp = max(delaysSamp);

end


function [rxLLTF, H52TrueCFR] = runChannelAndGetTrueCFR( ...
    chan, txLLTF, maxDelaySamp, initTime, P)

chanInfoBefore = info(chan);

if isfield(chanInfoBefore, 'ChannelFilterDelay')
    chDelay = chanInfoBefore.ChannelFilterDelay;
else
    chDelay = 0;
end

tailPad = chDelay + maxDelaySamp;
txPad = [txLLTF; zeros(tailPad, 1)];

% Path gains are returned because PathGainsOutputPort = true.
[rxPad, pathGains] = chan(txPad, initTime);

chanInfoAfter = info(chan);

% Remove internal channel-filter delay and keep original L-LTF length.
rxLLTF = rxPad(chDelay + (1:numel(txLLTF)));

% True label: channel frequency response
H52TrueCFR = trueLLTFCFRFromPathGains(pathGains, chanInfoAfter, P);

release(chan);

end


function H52True = trueLLTFCFRFromPathGains(pathGains, chanInfo, P)
% Return true channel frequency response on the 52 active L-LTF tones.
%
% L-LTF CBW20 structure:
%   samples 1:32    = GI2
%   samples 33:96   = first useful long training symbol
%   samples 97:160  = second useful long training symbol
%
% Since X uses wlanLLTFDemodulate(..., symOffset=1.0), the FFT windows are
% aligned after GI2. Therefore, use only samples 33:160 and set cplen = 0
% for ofdmChannelResponse.

nfft = P.nfft;
activeFFTIdx = P.ofdmInfo.ActiveFFTIndices(:);
activeFreqIdx = P.ofdmInfo.ActiveFrequencyIndices(:);

pathFilters = chanInfo.ChannelFilterCoefficients;

if isfield(chanInfo, 'ChannelFilterDelay')
    toffset = chanInfo.ChannelFilterDelay;
else
    toffset = 0;
end

assert(size(pathGains,1) >= max(P.lltfUsefulSampleIdx), ...
    'Not enough path gain samples to cover the L-LTF useful symbols.');

pathGainsUse = pathGains(P.lltfUsefulSampleIdx,:,:,:);

if P.useOfdmChannelResponse
    % cplen = 0 because pathGainsUse excludes GI2 and contains only
    % two useful 64-sample L-LTF symbols.
    h = ofdmChannelResponse(pathGainsUse, pathFilters, nfft, 0, ...
        activeFFTIdx, toffset);

    % SISO: h is 52 x 2, or 52 x 2 x 1 x 1.
    h = h(:,:,1,1);

    % Match the L-LTF LS estimator convention: average two L-LTF symbols.
    H52True = mean(h, 2);
    H52True = H52True(:);
else
    % Fallback for MATLAB releases without ofdmChannelResponse.
    % This reconstructs the CFR from path gains and path filter coefficients.
    Hsym = zeros(numel(activeFreqIdx), 2);

    for isym = 1:2
        idx = (isym-1)*nfft + (1:nfft);
        Hsym(:,isym) = manualCFRFromPathGains( ...
            pathGainsUse(idx,:,:,:), pathFilters, ...
            activeFreqIdx, nfft, toffset);
    end

    H52True = mean(Hsym, 2);
    H52True = H52True(:);
end

assert(numel(H52True) == P.numTones, ...
    'Unexpected number of true CFR tones.');

end


function H = manualCFRFromPathGains(pathGainsSym, pathFilters, ...
    activeFreqIdx, nfft, toffset)
% Manual fallback for true CFR calculation.
%
% pathGainsSym:
%   Nfft-by-Npath path gain samples for one useful OFDM symbol.
%
% pathFilters:
%   Npath-by-Nh channel filter coefficients.
%
% activeFreqIdx:
%   WLAN active subcarrier frequency indices, e.g. [-26:-1, 1:26].'
%
% toffset:
%   channel filter delay removed at the receiver. Removing this delay
%   introduces exp(+j*2*pi*k*toffset/Nfft) in frequency.

numSamples = size(pathGainsSym, 1);
assert(numSamples == nfft, ...
    'manualCFRFromPathGains expects exactly one useful OFDM symbol.');

pg = reshape(pathGainsSym, size(pathGainsSym,1), []);
numPaths = size(pg, 2);

% MATLAB documentation specifies pathFilters as Np-by-Nh.
% This transpose guard handles older/object-specific orientation safely.
if size(pathFilters,1) ~= numPaths && size(pathFilters,2) == numPaths
    pathFilters = pathFilters.';
end

assert(size(pathFilters,1) == numPaths, ...
    'Number of path filters must match number of path gains.');

meanPathGains = mean(pg, 1);       % 1-by-Npath
hTaps = meanPathGains * pathFilters; % 1-by-Nh

tapIdx = 0:(numel(hTaps)-1);
activeFreqIdx = activeFreqIdx(:);

E = exp(-1j * 2*pi/nfft * (activeFreqIdx * tapIdx));
Hraw = E * hTaps(:);

timingPhase = exp(1j * 2*pi/nfft * activeFreqIdx * toffset);

H = Hraw .* timingPhase;
H = H(:);

end


function delaysSamp = sampleDelayProfile(numTaps, maxDelaySamples)

if numTaps == 1
    delaysSamp = 0;
    return;
end

cand = 1:maxDelaySamples;
weights = exp(-cand / max(1, maxDelaySamples/4));

delays = zeros(1, numTaps - 1);
available = cand;
availableW = weights;

for ii = 1:(numTaps - 1)
    p = availableW / sum(availableW);
    u = rand;
    idx = find(cumsum(p) >= u, 1, 'first');

    delays(ii) = available(idx);

    available(idx) = [];
    availableW(idx) = [];
end

delaysSamp = sort([0, delays]);

end


function x = sampleTruncGaussian(mu, sigma, xmin, xmax)

while true
    x = mu + sigma * randn;

    if x >= xmin && x <= xmax
        return;
    end
end

end


function y = addAwgnBySNR(x, snrDb)
% Add AWGN using Communications Toolbox awgn.
% The 'measured' option estimates signal power from x.

if isinf(snrDb)
    y = x;
    return;
end

y = awgn(x, snrDb, 'measured');

end


function row = complexToIQIQRow(x)
x = x(:).';

row = zeros(1, 2*numel(x), 'single');
row(1:2:end) = single(real(x));
row(2:2:end) = single(imag(x));

end
