function generate_wifi_lltf_dataset(snrDb, channelType)
% Generate two datasets with the same SNR simulation process:
% 1) 100000 samples -> <baseName>.csv
% 2) 10000 samples  -> <baseName>_eval.csv
%
% channelType:
%   'onetap'   -> one-tap flat Rayleigh block fading
%   'rayleigh' -> multipath Rayleigh fading channel
%   'rician'   -> multipath Rician fading channel
%
% useCFO:
%   true  -> apply CFO
%   false -> remove CFO effect

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
numEvalSamples = 10000;

%% Output files
% Keep original naming so existing wrapper stays compatible.
baseName = sprintf('wifi_lltf_dataset_%ddb', round(snrDb));
mainFile = sprintf('%s.csv', baseName);
evalFile = sprintf('%s_eval.csv', baseName);

%% Parameters
cbw = 'CBW20';
fcHz = 2.412e9;
useCFO = false;

%% Channel Setting
% Used for multipath Rayleigh / Rician
numTapsRange = [3 5];
maxDelaySamples = 8;
pdpTauSamples = 1.5;

% Used for Rician
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

% CFO setting
cfoPpmMean = 0;
cfoPpmStd  = 1;
cfoPpmClip = 3;

% Seeds
mainSeed = 94;
evalSeed = 3094;

%% Checks
assert(isscalar(snrDb) && isfinite(snrDb), 'snrDb must be a finite scalar.');
assert(numTapsRange(1) >= 1, 'numTapsRange(1) must be >= 1.');
assert(numTapsRange(1) <= numTapsRange(2), 'Invalid numTapsRange.');
assert(speedRangeMps(1) <= speedRangeMps(2), 'Invalid speedRangeMps.');

%% WLAN setup
cfg = wlanNonHTConfig('ChannelBandwidth', cbw);
Fs = wlanSampleRate(cfg);
cLight = 299792458;
lambda = cLight / fcHz;

txLLTF = wlanLLTF(cfg);
txLLTF = txLLTF(:);

ofdmInfo = wlanNonHTOFDMInfo('L-LTF', cbw);
numTones = ofdmInfo.NumTones;

assert(numel(txLLTF) == 160, 'Expected 160 complex L-LTF samples for CBW20.');
assert(numTones == 52, 'Expected 52 active tones for CBW20 L-LTF.');

numInputCols = 160 * 2;
numLabelCols = 52 * 2;
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
P.cfoPpmMean = cfoPpmMean;
P.cfoPpmStd = cfoPpmStd;
P.cfoPpmClip = cfoPpmClip;
P.useCFO = useCFO;

P.randomStartTimeMaxSec = randomStartTimeMaxSec;

%% Summary
fprintf('Configuration summary\n');
fprintf('  Channel type     : %s\n', channelType);
fprintf('  Bandwidth        : %s\n', cbw);
fprintf('  Sample rate      : %.3f MHz\n', Fs/1e6);
fprintf('  Center frequency : %.6f GHz\n', fcHz/1e9);
fprintf('  SNR (dB)         : %.1f\n', snrDb);
fprintf('  Use CFO          : %d\n', useCFO);
fprintf('  Main samples     : %d\n', numMainSamples);
fprintf('  Eval samples     : %d\n', numEvalSamples);
fprintf('  Main file        : %s\n', mainFile);
fprintf('  Eval file        : %s\n\n', evalFile);

%% Generate main dataset
mainData = generateOneDataset(numMainSamples, snrDb, P, mainSeed, 'main');
writematrix(mainData, mainFile);

%% Generate eval dataset
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
    %% 1) SNR + CFO
    thisSnrDb = snrDb;

    if P.useCFO
        rawCfoPpm = P.cfoPpmMean + P.cfoPpmStd * randn;
        rawCfoPpm = min(max(rawCfoPpm, -P.cfoPpmClip), P.cfoPpmClip);
        thisCfoHz = rawCfoPpm * 1e-6 * P.fcHz;
    else
        thisCfoHz = 0;
    end

    %% 2) Clean received L-LTF after selected channel
    rxLLTF = passThroughSelectedChannel(P, baseSeed, n);

    %% 3) Label = frequency-domain channel response on 52 active tones
    demodLLTF = wlanLLTFDemodulate(rxLLTF, P.cfg);
    chEst = wlanLLTFChannelEstimate(demodLLTF, P.cfg);
    H52 = chEst(:,1,1);

    %% 4) Input X = channel + CFO + AWGN
    rxLLTFWithCFO = applyCFO(rxLLTF, thisCfoHz, P.Fs);
    rxLLTFIn = addAwgnBySNR(rxLLTFWithCFO, thisSnrDb);

    %% 5) Save row
    DATA(n,:) = single([complexToIQIQRow(rxLLTFIn), complexToIQIQRow(H52)]);

    if mod(n, 5000) == 0 || n == numSamples
        fprintf('  %6d / %6d done\n', n, numSamples);
    end
end

elapsedSec = toc;
fprintf('Finished %s dataset in %.2f sec\n\n', tag, elapsedSec);

end


function rxLLTF = passThroughSelectedChannel(P, baseSeed, sampleIdx)
switch P.channelType
    case 'onetap'
        seedNow = baseSeed + sampleIdx - 1;
        chan = createStdOneTapRayleighChannel(P.Fs, seedNow);
        rxLLTF = runChannel(chan, P.txLLTF, 0, 0);

    case 'rayleigh'
        [pathDelaysSec, avgPathGainsDb, maxDelaySamp, thisMaxFdHz, seedNow, initTime] = ...
            sampleChannelProfile(P, baseSeed, sampleIdx);

        chan = createStdRayleighChannel(P.Fs, pathDelaysSec, avgPathGainsDb, ...
            thisMaxFdHz, seedNow);

        rxLLTF = runChannel(chan, P.txLLTF, maxDelaySamp, initTime);

    case 'rician'
        [pathDelaysSec, avgPathGainsDb, maxDelaySamp, thisMaxFdHz, seedNow, initTime, numTaps] = ...
            sampleChannelProfile(P, baseSeed, sampleIdx);

        % KFactor must be linear scale, not dB.
        kVec = zeros(1, numTaps);
        kVec(1) = 10^(sampleTruncGaussian(P.kDbMean, P.kDbStd, ...
            P.kDbMin, P.kDbMax) / 10);

        % Optional weak LOS on second path.
        if numTaps >= 2 && rand < P.pSecondWeakRician
            kVec(2) = 10^(sampleTruncGaussian(P.kDbMeanWeak, P.kDbStdWeak, ...
                P.kDbMinWeak, P.kDbMaxWeak) / 10);
        end

        % LOS parameters only for paths with positive KFactor.
        losFd = zeros(1, numTaps);
        losPhase = zeros(1, numTaps);
        idxLos = (kVec > 0);
        losFd(idxLos) = (2 * rand(1, nnz(idxLos)) - 1) * thisMaxFdHz;
        losPhase(idxLos) = 2 * pi * rand(1, nnz(idxLos));

        chan = createStdRicianChannel(P.Fs, pathDelaysSec, avgPathGainsDb, ...
            kVec, losFd, losPhase, thisMaxFdHz, seedNow);

        rxLLTF = runChannel(chan, P.txLLTF, maxDelaySamp, initTime);

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
    'Seed', seedNow);
end


function chan = createStdRayleighChannel(Fs, pathDelaysSec, avgPathGainsDb, ...
    maxFdHz, seedNow)

chan = comm.RayleighChannel( ...
    'SampleRate', Fs, ...
    'PathDelays', pathDelaysSec, ...
    'AveragePathGains', avgPathGainsDb, ...
    'NormalizePathGains', true, ...
    'MaximumDopplerShift', maxFdHz, ...
    'FadingTechnique', 'Sum of sinusoids', ...
    'InitialTimeSource', 'Input port', ...
    'RandomStream', 'mt19937ar with seed', ...
    'Seed', seedNow);
end


function chan = createStdRicianChannel(Fs, pathDelaysSec, avgPathGainsDb, ...
    kVec, losFd, losPhase, maxFdHz, seedNow)

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
    'Seed', seedNow);
end


function [pathDelaysSec, avgPathGainsDb, maxDelaySamp, thisMaxFdHz, seedNow, initTime, numTaps] = ...
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


function rxLLTF = runChannel(chan, txLLTF, maxDelaySamp, initTime)
chanInfo = info(chan);
if isfield(chanInfo, 'ChannelFilterDelay')
    chDelay = chanInfo.ChannelFilterDelay;
else
    chDelay = 0;
end

tailPad = chDelay + maxDelaySamp;
txPad = [txLLTF; zeros(tailPad, 1)];

rxPad = chan(txPad, initTime);
rxLLTF = rxPad(chDelay + (1:numel(txLLTF)));

release(chan);
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


function y = applyCFO(x, cfoHz, Fs)
if cfoHz == 0
    y = x;
    return;
end

n = (0:numel(x)-1).';
y = x .* exp(1j * 2 * pi * cfoHz * n / Fs);
end


function y = addAwgnBySNR(x, snrDb)
if isinf(snrDb)
    y = x;
    return;
end

signalPower = mean(abs(x).^2);
noisePower = signalPower / (10^(snrDb / 10));
noise = sqrt(noisePower / 2) * (randn(size(x)) + 1j * randn(size(x)));
y = x + noise;
end


function row = complexToIQIQRow(x)
x = x(:).';
row = zeros(1, 2 * numel(x), 'single');
row(1:2:end) = single(real(x));
row(2:2:end) = single(imag(x));
end
