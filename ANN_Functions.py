

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import timeit

class fun(object):
	def __init__(self, filename):
		self.tokenList = open(filename, 'r')

	def initialize_starting_weights(WEIGHTS, n_input, n_classes, archit, layer_number, df, mu, sigma):
		"""
		Initializes starting weights and biases, with weights determined by:
			- random	Standard normal distribution 
			- xavier 	Useful when # features is large (Glorot and Yoshua (2010))
			- file 		Use weights from input file with noise infusion (options: mu, sigma)
		"""
		weights = {}
		biases = {}

		if WEIGHTS == 'xavier':
			initializer = tf.contrib.layers.xavier_initializer()
			weights['h1'] = tf.Variable(initializer([n_input, archit[0]]))
			biases['b1'] = tf.Variable(tf.random_normal([archit[0]]))
			for l in range(1,layer_number):
				w_name = 'h' + str(l+1)
				b_name = 'b' + str(l+1)
				weights[w_name] = tf.Variable(initializer([archit[l-1], archit[l]]))
				biases[b_name] = tf.Variable(tf.random_normal([archit[l]]))
			weights['out'] = tf.Variable(initializer([archit[-1], n_classes]))
			biases['out'] = tf.Variable(tf.random_normal([n_classes]))

		elif WEIGHTS == 'random':
			weights['h1'] = tf.Variable(tf.random_normal([n_input, archit[0]]))
			biases['b1'] = tf.Variable(tf.random_normal([archit[0]]))
			for l in range(1,layer_number):
				w_name = 'h' + str(l+1)
				b_name = 'b' + str(l+1)
				weights[w_name] = tf.Variable(tf.random_normal([archit[l-1], archit[l]]))
				biases[b_name] = tf.Variable(tf.random_normal([archit[l]]))
			weights['out'] = tf.Variable(tf.random_normal([archit[-1], n_classes]))
			biases['out'] = tf.Variable(tf.random_normal([n_classes]))
		else:
			print('Seeding starting weights using: %s + noise infusion (mu = %f, sigma= %f)' % (WEIGHTS, mu, sigma))
			w_file = pd.read_csv(WEIGHTS, sep=',', index_col = None, header=0, names=['ID', 'weight'])
			w_file = w_file[w_file['ID'].isin(list(df))] # Drop weights not included in x
			w_file.ID = w_file.ID.astype('category') # Convert to category 
			w_file.ID.cat.set_categories(list(df), inplace=True) # Set sorter for categories as X order
			w_file = w_file.sort_values(['ID'])
			weights = w_file['weight']
			weight_df = pd.concat([weights]*archit[0], axis=1, ignore_index=True)
			noise = np.random.normal(mu, sigma, [n_input, archit[0]])
			weights_noise = weight_df + noise
			weights_noise = weights_noise.astype(np.float32)
			weights['h1'] = tf.Variable(weights_noise)

		return weights, biases

	def define_architecture(arc):
		# Define Architecture
		try:
			hidden_units = arc.strip().split(',')
			archit = list(map(int, hidden_units))
		except:
			archit = [arc]
		layer_number = len(archit)

		return archit, layer_number
	
	def define_loss(loss_type, nn_y, pred, l2, weights):
		"""Define loss function, taking into account L2 penalty if needed"""
		
		if loss_type == 'mse':
	 		loss = tf.losses.mean_squared_error(nn_y, pred)
	 		if l2 != 0.0:
	 			try:
	 				regularizer = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2'])
	 			except:
	 				regularizer = tf.nn.l2_loss(weights['h1'])
	 			loss = tf.reduce_mean(loss + l2 * regularizer)
	 		else:
	 			loss = tf.reduce_mean(loss)
		else:
	 		print('That loss function is not implemented...')
	 		quit()

		return loss

	def multilayer_perceptron(x, weights, biases, layer_number, activation_function, dropout, dropout_rate):
		"""
		Generate MLP model of any shape with options for activation function type and dropout levels
		currently: sigmoid, relu, elu
		"""
		layer = x
		for l in range(1,layer_number+1):
			weight_name = 'h' + str(l)
			bias_name = 'b' + str(l)
			layer = tf.add(tf.matmul(layer, weights[weight_name]), biases[bias_name])
			if activation_function.lower() == 'sigmoid':
				layer = tf.nn.sigmoid(layer)
			elif activation_function.lower() == 'relu':
				layer = tf.nn.relu(layer)
			elif activation_function.lower() == 'elu':
				layer = tf.nn.elu(layer)
			else:
				print("Given activation function is not supported")
				quit()
			if dropout != 0:
				dropout_rate = 1.0 - dropout
				drop_out = tf.nn.dropout(layer, dropout_rate)
		out_layer = tf.matmul(layer, weights['out']) + biases['out']
		return out_layer

	def save_trained_weights(SAVE, tvars, tvars_vals, archit, feat_list):


		for var, val in zip(tvars, tvars_vals):
			if var.name == 'Variable:0':
				col_names = ['Node_1_%s' % s for s in range(1,archit[0]+1)] 
				weights_l1 = pd.DataFrame(val, index=feat_list, columns=col_names)
				print('\n\n\nSnapshot of weights in first hidden layer:')
				print(weights_l1.head())

		if len(archit) == 1:
			for var, val in zip(tvars, tvars_vals):
				if var.name == 'Variable_2:0':
					weights_final = pd.DataFrame(val, index=list(weights_l1), columns=['Final'])

		elif len(archit) == 2:
			for var, val in zip(tvars, tvars_vals):
				if var.name == 'Variable_2:0':
					col_names = ['Node_2_%s' % s for s in range(1,archit[1]+1)] 
					weights_l2 = pd.DataFrame(val, index=list(weights_l1), columns=col_names )
					weights_l2.to_csv(SAVE+'_Weights_HL2.csv', index=True)
				elif var.name == 'Variable_4:0':
					weights_final = pd.DataFrame(val, index=list(weights_l2), columns=['Final'])

		elif len(archit) == 3:
			for var, val in zip(tvars, tvars_vals):
				if var.name == 'Variable_2:0':
					col_names = ['Node_2_%s' % s for s in range(1,archit[1]+1)] 
					weights_l2 = pd.DataFrame(val, index=list(weights_l1), columns=col_names )
					weights_l2.to_csv(SAVE+'_Weights_HL2.csv', index=True)
				elif var.name == 'Variable_4:0':
					col_names = ['Node_3_%s' % s for s in range(1,archit[2]+1)] 
					weights_l3 = pd.DataFrame(val, index=list(weights_l2), columns=col_names )
					weights_l3.to_csv(SAVE+'_Weights_HL3.csv', index=True)
				elif var.name == 'Variable_6:0':
					weights_final = pd.DataFrame(val, index=list(weights_l3), columns=['Final'])

		weights_l1.to_csv(SAVE+'_Weights_HL1.csv', index=True)
		weights_final.to_csv(SAVE+'_Weights_Fin.csv', index=True)

