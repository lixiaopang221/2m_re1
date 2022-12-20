# KID_T_2m_re1
### This is an implementation of the paper Cross-Modality Deep Feature Learning for Brain Tumor Segmentation published on Pattern Recognition. 
### Environment
---
python(Tested on 3.8)  
pytorch(Tested on 1.13.0)  
### Data Set
---    
KiTS19  data set(including training set and validation set).   
Sample list file `*.txt`(eg. train.txt), which contains the filenames of the inputs.   
### Preprocessing
--- 
Before training or testing, a preprocessing is needed to generate proper data format for the implementation.  
* training data 
```
python run_preprocessing_KidT.py --mode=train
``` 
* validation data 
```
python run_preprocessing_KidT.py --mode=val
``` 

### Test
---
Change the "run_mode" to "test" in the code file `seg2g.py` and run:  
``` 
python seg2g_KidT_test.py
```
### Train
---  
The training of this implementation is divided into two stages. The first stage is to train CycleGAN network, and the second stage is to train the segmentation network based on transfer learning.
* The first stage
``` 
python gan_KidT.py
```
* The second stage, set the run the "run_mode" to "train" and run:
```
python seg2g_KidT.py
```
