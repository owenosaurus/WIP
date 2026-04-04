function resultsTbl = run_wifi_lltf_snr_sweep_wrapper()
% Call external functions:
%   1) generate_wifi_lltf_dataset(snrDb)
%   2) [lsMae, lmmseMae] = evaluate_wifi_lltf_ls_mmse_mae(evalCsv)
%
% SNR sweep: 18, 15, 12, 9, 6, 3, 0 dB

snrList = 18:-3:0;

lsMaeList = zeros(numel(snrList), 1);
lmmseMaeList = zeros(numel(snrList), 1);

for ii = 1:numel(snrList)
    snrDb = snrList(ii);

    fprintf('============================================================\n');
    fprintf('SNR = %d dB\n', snrDb);

    %generate_wifi_lltf_dataset(snrDb);

    evalCsv = sprintf('wifi_lltf_dataset_%ddb_eval.csv', round(snrDb));
    [lsMae, lmmseMae] = evaluate_wifi_lltf_ls_mmse_mae(evalCsv);

    lsMaeList(ii) = lsMae;
    lmmseMaeList(ii) = lmmseMae;
end

resultsTbl = table(snrList(:), lsMaeList, lmmseMaeList, ...
    'VariableNames', {'snrDb', 'lsMae', 'lmmseMae'});

csvFile = 'wifi_lltf_results.csv';
writetable(resultsTbl, csvFile);

fprintf('============================================================\n');
fprintf('Saved result table: %s\n', csvFile);

end
