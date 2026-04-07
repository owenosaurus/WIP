function resultsTbl = run_wifi_lltf_snr_sweep_wrapper()
% Call external functions:
%   1) generate_wifi_lltf_dataset(snrDb, channelType)
%   2) [lsNrmse, lmmseNrmse] = evaluate_wifi_lltf_ls_mmse_nrmse(evalCsv)
%
% SNR sweep: 0, 3, 6, 9, 12, 15, 18 dB

snrList = 0:3:18;

lsNrmseList = zeros(numel(snrList), 1);
lmmseNrmseList = zeros(numel(snrList), 1);

channelType = 'rician';   % 'awgn', 'rayleigh', or 'rician'

for ii = 1:numel(snrList)
    snrDb = snrList(ii);

    fprintf('============================================================\n');
    fprintf('SNR = %d dB\n', snrDb);
    fprintf('channelType = %s\n', channelType);

    generate_wifi_lltf_dataset(snrDb, channelType);

    evalCsv = sprintf('wifi_lltf_dataset_%ddb_eval.csv', round(snrDb));

    [lsNrmse, lmmseNrmse] = evaluate_wifi_lltf_ls_mmse_nrmse(evalCsv);

    lsNrmseList(ii) = lsNrmse;
    lmmseNrmseList(ii) = lmmseNrmse;
end

resultsTbl = table(snrList(:), lsNrmseList, lmmseNrmseList, ...
    'VariableNames', {'SNR_dB', 'LS_NRMSE', 'LMMSE_NRMSE'});

csvFile = sprintf('wifi_lltf_results_%s.csv', channelType);
writetable(resultsTbl, csvFile);
fprintf('Saved result table: %s\n', csvFile);

end
