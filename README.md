# Offense_det
Offense detection code.

How to run:
In order to replicate anchor paper results, run the following configuration: 
```
train.py 
-bs=24
-lr=1e-5
-ep=100
-pa=3
--model=xlm_r
--task=all
--clip
--cuda=1
-lw
0.6
0.2
0.2
``` 
The model class is : 
**Paper_recon** which is in models\project_models.py
In order to run our innovative model, run the following configuration: 
```
train.py 
-bs=24
-lr=1e-5
-ep=100
-pa=3
--model=bert
--task=all
--clip
--cuda=1
-lw
0.6
0.2
0.2
```  
The model class is :  
**ODF** which is in models\project_models.py

![alt text](https://github.com/YuvalBecker/Offense_det/blob/main/model_arch.png)

Figure 1, Our suggested architecture 
