import TENSORFLOW_MODEL
import UTIL
import time
import json

MODEL_EPOCH = 70
MODEL_WIDTH = 150
MODEL_X_DIMENSION = 300
MODEL_Y_DIMENSION = 300
MODEL_LEARNING_RATE = 0.002
MODEL_ALPHA = 0.9
KERNEL_NUM = 3
MODEL_BATCHSIZE = 2
testNum = 1


plotFilePath = "../plots_vec_wordbased/"
parameterPath = 'parameter_mini/'

LONGEST_P_NUM = 2392



print('File \'json\' Loading.....')
with open("../MovieQA/data/qa.mini_train.300d.json") as data_file:    
	train_q = json.load(data_file)

with open("../MovieQA/data/qa.mini_val.300d.json") as data_file:   
	val_q = json.load(data_file)





####  give P and several Q and choice and segment array  #####
print('Training start....')
for i in range(testNum):
	start_time = time.time()
	print("building model")
	model = TENSORFLOW_MODEL.MODEL(	MODEL_X_DIMENSION,
									MODEL_Y_DIMENSION,
									MODEL_WIDTH,
									MODEL_LEARNING_RATE,
									KERNEL_NUM,
									parameterPath,
									MODEL_BATCHSIZE,
									LONGEST_P_NUM
								)
	print("building finish")
	model.initialize()
	UTIL.Train_TestMODEL(	train_q,
							val_q,
							plotFilePath,
							model,
							MODEL_EPOCH,
							MODEL_BATCHSIZE,
							LONGEST_P_NUM
						)
	print("--- MODEL %s seconds ---" % (time.time() - start_time))

	
	
	
