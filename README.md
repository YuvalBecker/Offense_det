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
--hard_mine
False
--hard_mine_freq
-1
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
--hard_mine
False
--hard_mine_freq
-1
```  
The model class is :  
**ODF** which is in models\project_models.py

![alt text](https://github.com/YuvalBecker/Offense_det/blob/main/model_arch.png)

Figure 1, Our suggested architecture 


As explained in the report, there are several training steps to our algorithm.


**Step 1:** Train over all dataset in the most standard way, as can be obtained by the command above.


**Step 2:** Training over all dataset, while using a balnced distribution between the different tasks.


Add the additional fields:

```
--balance=True
--balancer_percent= 0.7
--ckpt="path to the previous best model"
```

This command every epoch has a probabillity of 0.7 to sample a uniform distribution of samples between the three tasks.

This command calls to **data_balancer** function which is in trainer.py


**Step 3:** Train using cyclic hard mining.


Add the additional fields:
```
--hard_mine=True
--hard_mine_freq = 3
```
This command, every three epochs updated the training dataset by collecting the most difficult samples for our current model on-going trained model. 

This command calls to **hard_collection** which is in \data.py 


