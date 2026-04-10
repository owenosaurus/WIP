function [lsNmae, lmmseNmae] = evaluate_wifi_lltf_ls_mmse_nmae(evalCsv)
% Evaluate LS and same-file MMSE using one *_eval.csv file.
%
% Dataset row format:
% [ X(1)_I X(1)_Q ... X(160)_I X(160)_Q  Y(1)_I Y(1)_Q ... Y(52)_I Y(52)_Q ]
%
% X : received L-LTF after channel + CFO + AWGN
% Y : ground-truth channel label H(52)
%
% Output:
%   lsNmae    : actually NMAE of LS estimate
%   lmmseNmae : actually NMAE of same-file MMSE estimate

if nargin < 1
    evalCsv = 'wifi_lltf_dataset_18db_eval.csv';
end

cfg = wlanNonHTConfig('ChannelBandwidth', 'CBW20');

[rxEval, hTrue] = readDataset(evalCsv);
numSamples = size(rxEval, 2);

fprintf('EVAL CSV: %s\n', evalCsv);

hLs = zeros(52, numSamples);

for n = 1:numSamples
    rx = rxEval(:, n);
    hLs(:, n) = estimateLS(rx, cfg);
end

%  NMAE
lsNmae = computeNMAE(hLs, hTrue);

% same-file channel covariance based MMSE
muH = mean(hTrue, 2);
Hc  = hTrue - muH;
Rhh = (Hc * Hc') / max(numSamples - 1, 1);

reg = 1e-8 * real(trace(Rhh)) / size(Rhh, 1);
Rhh = Rhh + reg * eye(size(Rhh));

hLmmse  = zeros(52, numSamples);
noiseVar = zeros(1, numSamples);

for n = 1:numSamples
    rx = rxEval(:, n);
    demodLLTF = wlanLLTFDemodulate(rx, cfg);
    noiseVar(n) = wlanLLTFNoiseEstimate(demodLLTF);
    
    % L-LTF LS error variance
    sigma2 = max(real(noiseVar(n)) / 2, 0);

    hLmmse(:, n) = muH + (Rhh / (Rhh + sigma2 * eye(size(Rhh)))) * (hLs(:, n) - muH);
end

lmmseNmae = computeNMAE(hLmmse, hTrue);

fprintf('Results\n');
fprintf('LS NMAE    : %.6f\n', lsNmae);
fprintf('MMSE NMAE  : %.6f\n\n', lmmseNmae);

end


function Hls = estimateLS(rxLLTF, cfg)
demodLLTF = wlanLLTFDemodulate(rxLLTF, cfg);
chEst = wlanLLTFChannelEstimate(demodLLTF, cfg);
Hls = chEst(:, 1, 1);
end


function nmae = computeNMAE(Hhat, Htrue)
num = sum(abs(Hhat(:) - Htrue(:)));
den = sum(abs(Htrue(:)));

assert(den > 0, 'Ground-truth magnitude sum is zero, cannot compute NMAE.');

nmae = num / den;
end


function [rxEval, hLabel] = readDataset(csvFile)
data = readmatrix(csvFile);

numColsExpected = 160 * 2 + 52 * 2;
assert(size(data, 2) == numColsExpected, ...
    'Unexpected number of columns in %s. Expected %d, got %d.', ...
    csvFile, numColsExpected, size(data, 2));

xPart = data(:, 1:320);
yPart = data(:, 321:424);

rxEval = complex(xPart(:, 1:2:end), xPart(:, 2:2:end)).';
hLabel = complex(yPart(:, 1:2:end), yPart(:, 2:2:end)).';
end
