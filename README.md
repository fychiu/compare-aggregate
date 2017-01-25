# compare-aggragate
compare-aggregate model on MovieQA task

## Preliminary:
```
1. Tensorflow   

2. 7zip   
sudo apt-get install p7zip-full   
7z x filename.7z
```

## To use model:

To train the compared-aggregate model, files should be placed into corresponding folders properly.
Three python scripts put up the comp-aggr model: TENSORFLOW_MODEL.py, UTIL.py, main.py
   
   
UTIL.py contains all the helper functions, including padding, batching and some preprocessing.
TENSORFLOW_MODEL.py constructs the model itself using Tensorflow.
main.py imports the above python scripts and starts running the model

In the command line, you can simply start training and testing with the command:    
```
python main.py
```
   
If all the needed files and data are placed properly, the training should run successfully.
In default, every 50 training batches will be followed by 1 testing(on validation set) batch to check the current progress. And every 15 epoch, we will perform testing on training set to examine overfitting.


   
## Directories:
```
comp-agg-model:
   three .py files
```

``` 
plot_vec_wordbased:
===> plot files in which the text is coverted to word vectors
```
``` 
MovieQA/split_data:
===> qa.json files in which the text is converted wo word vectors
===> there are some different size for qa files
1. qa.mini_train.split.json: the first 1000 data in training data
   qa.mini_val.split.json: the first 200 data in validation data
2. qa.train5000.split.json: the first 5000 data in training data
   qa.val1000.split.json: the first 1000 data in validation data
3. qa.train.split.json: the whole data in training data
   qa.val.split.json: the whole data in validation data
```
