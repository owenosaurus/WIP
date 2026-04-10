# DNN based WiFi Channel Estimaion (vanilla)

## MLP structure
input shape=(2, 64, 2)  
hidden dims=(128, 128, 128)  
output shape=(52, 2)  
adam optimizer    
Loss Function: MSE  
Evaluation Metrics: NMAE  

## Setting Chanel
Center Frequency = 2.412e9 GHz  
Channel Bandwidth = 20 MHz  
Sampling rate = 20 MHz  
CFO = 0  
SNR = [0, 3, 6, 9, 12, 15, 18] dB  

### onetap

### rayliegh

### rician

## dataset size
traing sample 100k  
evaluation sample 10k
