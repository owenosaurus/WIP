# WiFi Channel Estimaion

## MLP structure
input_shape=(2, 64, 2)  
hidden_dims=(256, 128, 128, 64)  
output_shape=(52, 2)  


## Channel setting
fcHz = 2.412e9  
numTapsRange = [2 8]  
maxDelaySamples = 20  
pdpTauSamples = 4  

pLOS = 0.60  

kDbMean = 7.0  
kDbStd  = 4.0  
kDbMin  = 0.0  
kDbMax  = 15.0  

pSecondWeakRician = 0.20  
kDbMeanWeak = 3.0  
kDbStdWeak  = 2.0  
kDbMinWeak  = 0.0  
kDbMaxWeak  = 6.0  

speedRangeMps = [0 10]  

## dataset size
traing 100000  
evaluation 10000
