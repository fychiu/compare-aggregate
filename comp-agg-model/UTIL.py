import numpy as np
from numpy import sum
from random import shuffle
#import operator
import json
from pprint import pprint
#import cPickle as pickle

import os, sys

def batchPadding(plot):
	num = 0
	filler = [0]*300
	for sentences in plot:
		if len(sentences)>=num:
			num = len(sentences)

	newPlot = []
	for sentences in plot:
		newPlot.append(sentences + [filler]*(num-len(sentences)))
		
	
	return newPlot

def Train_TestMODEL(train_q,val_q,plotFilePath,GRUAttention_object,epoch,batch_size,LONGEST_P_NUM) :	# train one epoch
	
	filler = [0]*300
	
	for i in range(epoch):

		print(' *** START TRAINING *** ')

		totalcost = 0
		count = 0


		batchP = []
		batchQ = []
		batchA = []
		batchB = []
		batchC = []
		batchD = []
		batchE = []
		batchAns = []
		for question in train_q:
			imdb_key = plotFilePath+question["imdb_key"]+".wiki.json"
			with open(imdb_key) as data_file:
				rawP = json.load(data_file)['plot_data']
			P = rawP + [filler]*(LONGEST_P_NUM-len(rawP))
			batchP.append(P)
			batchQ.append(question["question"])
			AnsVec = []
			AnsOption = []
			for j in range(5): 
				if question["answers"][j]:
					pass
				else:
					question["answers"][j] = [filler]
				AnsVec.append(question["answers"][j])
				AnsOption.append(0)
			AnsOption[question["correct_index"]] = 1
			batchAns.append(AnsOption)
			batchA.append(AnsVec[0])
			batchB.append(AnsVec[1])
			batchC.append(AnsVec[2])
			batchD.append(AnsVec[3])
			batchE.append(AnsVec[4])
			'''
			print(np.array(P).shape)
			print(np.array(Q).shape)
			print(np.array(AnsVec[0]).shape)
			print(np.array(AnsVec[1]).shape)
			print(np.array(AnsVec[2]).shape)
			print(np.array(AnsVec[3]).shape)
			print(np.array(AnsVec[4]).shape, type(AnsVec[4]))
			'''
			if len(batchP) == batch_size:
				count+=1
				batchP = np.asarray(batchP)
				batchQ = np.asarray(batchPadding(batchQ))
				batchA = np.asarray(batchPadding(batchA))
				batchB = np.asarray(batchPadding(batchB))
				batchC = np.asarray(batchPadding(batchC))
				batchD = np.asarray(batchPadding(batchD))
				batchE = np.asarray(batchPadding(batchE))
				batchAns = np.asarray(batchAns)
				cost,predict = GRUAttention_object.train(batchP,batchQ,batchA,batchB,batchC,batchD,batchE,batchAns)
				totalcost+=cost
				correct = 0
				for index,ans in enumerate(predict):
					correct += batchAns[index,ans]
				
				print('Cost of Epoch ' + str(i) + ' batch ' + str(count) + ": "+str(cost) + "  correct_num:" +str(correct))
				batchP = []
				batchQ = []
				batchA = []
				batchB = []
				batchC = []
				batchD = []
				batchE = []
				batchAns = []

		print('Total Cost of Epoch ' + str(i) +': ' + str(totalcost))
		count = 0
		batchP = []
		batchQ = []
		batchA = []
		batchB = []
		batchC = []
		batchD = []
		batchE = []
		batchAns = []
		
		#######    Test validation set   #########
		testcorrect = 0.0

		for question in val_q:
			imdb_key = plotFilePath+question["imdb_key"]+".wiki.json"
			with open(imdb_key) as data_file:
				rawTestP = json.load(data_file)['plot_data']
			P = rawTestP + [filler]*(LONGEST_P_NUM-len(rawTestP))
			batchP.append(P)
			batchQ.append(question["question"])
			AnsVec = []
			AnsOption = []
			for j in range(5): 
				if question["answers"][j]:
					pass
				else:
					question["answers"][j] = [filler]

				AnsVec.append(question["answers"][j])
				AnsOption.append(0)
			AnsOption[question["correct_index"]] = 1 
			batchAns.append(AnsOption)
			batchA.append(AnsVec[0])
			batchB.append(AnsVec[1])
			batchC.append(AnsVec[2])
			batchD.append(AnsVec[3])
			batchE.append(AnsVec[4])				

			if len(batchP) == batch_size:
				batchP = np.asarray(batchP)
				batchQ = np.asarray(batchPadding(batchQ))
				batchA = np.asarray(batchPadding(batchA))
				batchB = np.asarray(batchPadding(batchB))
				batchC = np.asarray(batchPadding(batchC))
				batchD = np.asarray(batchPadding(batchD))
				batchE = np.asarray(batchPadding(batchE))
				batchAns = np.asarray(batchAns)	
				predict = GRUAttention_object.predict(batchP,batchQ,batchA,batchB,batchC,batchD,batchE)
				
				for index,ans in enumerate(predict):
					correct += batchAns[index,ans]
				testcorrect += total


		print('correct:%d total:%d' % (testcorrect,len(val_q)))


		#######    Test training set   #########
		# traincorrect = 0.0
		# for question in train_q:
		# 	imdb_key = plotFilePath+question["imdb_key"]+".wiki.json"
		# 	with open(imdb_key) as data_file:
		# 		rawP = json.load(data_file)['plot_data']

		# 	P = rawP + [filler]*(LONGEST_P_NUM-len(rawP))
		# 	Q = question["question"]
		# 	AnsVec = []
		# 	AnsOption = []
		# 	for j in range(5): 
		# 		if question["answers"][j]:
		# 			pass
		# 		else:
		# 			question["answers"][j] = [filler]
				
		# 		AnsVec.append(question["answers"][j])
		# 		AnsOption.append(0)
		# 	AnsOption[question["correct_index"]] = 1 
			

			
		# 	tmp = GRUAttention_object.test(P,Q,AnsVec[0],AnsVec[1],AnsVec[2],AnsVec[3],AnsVec[4])



		# 	if AnsOption[tmp] == 1 :
		# 		traincorrect += 1
	 # 	print('correct:%d total:%d' % (traincorrect,len(train_q)))
	 # 	print('%f  %f' % (testcorrect/len(val_q),traincorrect/len(train_q))   )




def getLongestP(plotFilePath):

	longest = 0
	for plot in os.listdir(plotFilePath):
		if '.wiki.json' in plot:
			imdb_key = plotFilePath+plot
			with open(imdb_key) as data_file:
				P = json.load(data_file)['plot_data']
			lenP = len(P)
			if lenP >= longest :
				longest = lenP
	return longest

# def getLongestQA(train_q,val_q):
# 	longest_q = 0
# 	longest_ans = 0
# 	for question in train_q:
# 		if len(question["question"])>=longest_q:
# 			longest_q = len(question["question"])
# 		for ans in question["answers"]:
# 			if len(ans)>=longest_ans:
# 				longest_ans = len(ans)
# 	for question in val_q:
# 		if len(question["question"])>=longest_q:
# 			longest_q = len(question["question"])
# 		for ans in question["answers"]:
# 			if len(ans)>=longest_ans:
# 				longest_ans = len(ans)

# 	return longest_q,longest_ans






