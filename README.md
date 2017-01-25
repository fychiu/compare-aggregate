# compare-aggragate
compare-aggregate model on MovieQA task

To use the model:

1. Tensorflow

2. 7zip
   sudo apt-get install p7zip-full
   7z x filename.7z
   
   
comp-agg-model:


plot_vec_wordbased:
===> plot files in which the text is coverted to word vectors

MovieQA/split_data\n
===> qa.json files in which the text is converted wo word vectors\n
===> there are some different size for qa files\n
1. qa.mini_train.split.json: the first 1000 data in training data\n
   qa.mini_val.split.json: the first 200 data in validation data\n
2. qa.train5000.split.json: the first 5000 data in training data\n
   qa.val1000.split.json: the first 1000 data in validation data\n
3. qa.train.split.json: the whole data in training data\n
   qa.val.split.json: the whole data in validation data\n
