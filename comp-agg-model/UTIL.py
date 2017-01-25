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
