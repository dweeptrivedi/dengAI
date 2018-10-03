import os
import sys
import math
import time
import pandas as pd
import numpy as np
import torch
import torch.nn


import torch.utils.data
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def load_test_data():
	df_feat = pd.read_csv("dengue_features_test.csv",sep=',', index_col=False)
	
	feature_list = list(df_feat.columns)[4:]

	df_feat2 = df_feat[:][feature_list]

	df_feat2 = df_feat2.fillna(0).dropna(how="any")
	scaler = StandardScaler()
	df_feat2 = scaler.fit_transform(df_feat2.values)

	return df_feat2,df_feat


def load_data():
	df_feat = pd.read_csv("dengue_features_train.csv",sep=',', index_col=False)
	#print(df_feat.head())
	df_label = pd.read_csv("dengue_labels_train.csv",sep=',', index_col=False)
	#print(df_label.head())
	
	label_list = list(df_label.columns)[3:]
	feature_list = list(df_feat.columns)[4:]

	df_feat = df_feat[:][feature_list]
	df_label = df_label[:][label_list]

	df_feat = df_feat.fillna(0)
	scaler = StandardScaler()
	df_feat = scaler.fit_transform(df_feat.values)


	return df_feat, df_label.values




def main():

	dtype = torch.float
	device = torch.device("cpu")

	X, y = load_data()
	X = torch.from_numpy(X).float()
	y = torch.from_numpy(y).float()

	

	#graph dimensions
	D_in, H1, H2, H3, H4, H5, D_out = X.shape[1], 20, 20, 20, 20, 20, 1

	#graph
	model = torch.nn.Sequential(
		torch.nn.Linear(D_in, H1),
		torch.nn.LeakyReLU(),
		torch.nn.Linear(H1, H2),
		torch.nn.LeakyReLU(),
		torch.nn.Linear(H2, H3),
		torch.nn.LeakyReLU(),
		torch.nn.Linear(H3, H4),
		torch.nn.LeakyReLU(),
		torch.nn.Linear(H4, H5),
		torch.nn.LeakyReLU(),
		torch.nn.Linear(H5, D_out)
		)

	#loss functiom
	loss_fn = torch.nn.L1Loss(reduction='elementwise_mean')

	#optimizer
	lr = 1e-2
	optim = torch.optim.Adam(model.parameters(), lr=lr)

	dataset = torch.utils.data.TensorDataset(X, y)
	loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

	epoch = 1500
	avg_loss = 0
	track_avg = []

	for e in range(epoch):
		count =1 
		for X_train, y_train in loader:

			#forward pass
			y_pred = model(X_train)
			#loss
			loss = torch.sqrt(loss_fn(y_pred, y_train))
			avg_loss += loss.item()
			#print("{} batch_loss:{}".format(count,loss.item()))
			count+=1
			#initialize gradients to zero
			optim.zero_grad()

			#backward pass
			loss.backward()

			#update gradients
			optim.step()
		

		avg_epoch_loss = avg_loss/((e+1)*X.shape[0])
		print("epoch {} average loss:{}".format(e+1,avg_epoch_loss))
		track_avg.append(avg_epoch_loss)
		count+=1


	X, y = load_test_data()
	X = torch.from_numpy(X).float()

	y_pred = model(X)
	np.savetxt('pred.txt',y_pred.data.numpy().round(),delimiter=',',fmt='%d')



if __name__=="__main__":
	main()
