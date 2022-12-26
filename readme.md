### This is an implementation of the paper Cross-Modality Deep Feature Learning for Brain Tumor Segmentation published on Pattern Recognition.  The "2m_re1" program is for BraTS 2018. 
### Environment
---
python(Tested on 3.6)  
pytorch(Tested on 0.4.1)  
### Data Set
---  
BraTS 2017  data set(including training set and validation set).   
BraTS 2018  data set(including training set and validation set).   
Sample list file `*.txt`(eg. train.txt), which contains the filenames of the inputs.   
### Preprocessing
--- 
Before training or testing, a preprocessing is needed to generate proper data format for the implementation.  
* training data 
```
python run_preprocessing.py --mode=train
``` 
* validation data 
```
python run_preprocessing.py --mode=val
``` 

### Test
---
Change the "run_mode" to "test" in the code file `seg2g.py` and run:  
``` 
python seg2g.py
```
### Post processing
---
Change the file directories to your local data directories in the code file `seg_post.py` and run:
```
python seg_post.py
```
### Train
---  
The training of this implementation is divided into two stages. The first stage is to train CycleGAN network, and the second stage is to train the segmentation network based on transfer learning.
* The first stage
``` 
python gan.py
```
* The second stage, set the run the "run_mode" to "train" and run:
```
python seg2g.py
```
