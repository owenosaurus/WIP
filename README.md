# DNN based WiFi Channel Estimaion (vanilla)

## MLP structure
input shape=(2, 64, 2)  
hidden dimension=(128, 128, 128)  
output shape=(52, 2)  
adam optimizer    
Loss Function: MSE  
Evaluation Metrics: NMAE  

## Setting Chanel
Center Frequency = 2.412e9 GHz  
Channel Bandwidth = 20 MHz  
Sampling rate = 20 MHz  
SNR = [0, 3, 6, 9, 12, 15, 18] dB  

### onetap
AWGN

### rayliegh
numTapsRange = [3 5];
maxDelaySamples = 8;
pdpTauSamples = 1.5;  
speedRangeMps = [0 5];  
randomStartTimeMaxSec = 0.5;  

### rician
numTapsRange = [3 5];
maxDelaySamples = 8;
pdpTauSamples = 1.5;  

kDbMean = 3.0;  
kDbStd  = 2.0;  
kDbMin  = 0.0;  
kDbMax  = 7.0;  

pSecondWeakRician = 0.10;  
kDbMeanWeak = 1.5;  
kDbStdWeak  = 1.0;  
kDbMinWeak  = 0.0;  
kDbMaxWeak  = 4.0;  
speedRangeMps = [0 5];  
randomStartTimeMaxSec = 0.5;  

## dataset size
traing sample 100k  
evaluation sample 10k
