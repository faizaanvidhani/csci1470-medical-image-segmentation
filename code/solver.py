import os
import numpy as np
import time
import datetime
import tensorflow as tf
#import torch
#import torchvision
from tensorflow.keras import optimizers
#from torch.autograd import Variable
#import torch.nn.functional as F
import tensorflow.keras.activations as activations
from evaluation import *
from network2 import R2U_Net
from PIL import Image
from preprocess import tensor_to_image
import csv


class Solver(object):
	def __init__(self, train_inputs, train_labels, test_inputs, test_labels):

		# Data loader
		self.train_inputs = train_inputs
		self.train_labels = train_labels
		self.test_inputs = test_inputs 
		self.test_labels = test_labels 

		# Hyper-parameters
		self.lr = 0.0002
		self.beta1 = 0.5
		self.beta2 = 0.999

		# Model
		self.img_ch = 3
		self.output_ch = 1
		self.t = 3
		
		self.unet = R2U_Net(self.img_ch,self.output_ch,t=self.t)
		self.optimizer = optimizers.Adam(self.lr, self.beta1, self.beta2)
		
		self.augmentation_prob = 0.4

		# Training settings
		self.num_epochs = 1
		self.num_epochs_decay = 70
		self.batch_size = 10


		# Step size
		self.log_step = 2
		self.val_step = 2

		self.device = tf.device('cpu')
		#self.unet.to(self.device)
		self.model_type = 'R2U_Net'
		
	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img


	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
		
		# Train for Encoder
		
		for epoch in range(self.num_epochs):

			epoch_loss = 0
			
			acc = 0.	# Accuracy
			SE = 0.		# Sensitivity (Recall)
			SP = 0.		# Specificity
			PC = 0. 	# Precision
			F1 = 0.		# F1 Score
			JS = 0.		# Jaccard Similarity
			DC = 0.		# Dice Coefficient
			length = 0

			for i in range(0,len(self.train_inputs),self.batch_size):
				# GT : Ground Truth

				images = self.train_inputs[i:i+self.batch_size]
				#print('Images:  ', images)
				GT = self.train_labels[i: i+self.batch_size]

				# Backprop + optimize
				with tf.GradientTape() as tape:
					# SR : Segmentation Result
					SR = self.unet(images)
					#print("SR", SR)
					#SR_probs = activations.sigmoid(SR)
					SR_probs = activations.softmax(SR)
					#print("SR probs", SR_probs)
					SR_flat = tf.reshape(SR_probs, [tf.shape(SR_probs)[0],-1])
					GT_flat = tf.reshape(GT, [tf.shape(GT)[0], -1])
					#print("SR_flat Shape:", tf.shape(SR_flat))
					#print("GT_flat Shape:", tf.shape(GT_flat))
					#loss = tf.keras.metrics.binary_crossentropy(GT_flat, SR_flat)
					bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
					loss = bce(GT_flat, SR_flat)
					#loss = tf.keras.metrics.binary_crossentropy(GT, SR_probs)
					avg_loss = tf.reduce_mean(loss)
					print("loss is", avg_loss)
					epoch_loss += avg_loss

				gradients = tape.gradient(avg_loss, self.unet.trainable_variables)
				self.optimizer.apply_gradients(zip(gradients, self.unet.trainable_variables))

				acc += get_accuracy(SR,GT)
				""" SE += get_sensitivity(SR,GT)
				SP += get_specificity(SR,GT)
				PC += get_precision(SR,GT)
				F1 += get_F1(SR,GT)
				JS += get_JS(SR,GT)
				DC += get_DC(SR,GT) """
				length += 1

			acc = acc/length
			print("accuracy is", acc)
			print("length is", length)
			""" SE = SE/length
			SP = SP/length
			PC = PC/length
			F1 = F1/length
			JS = JS/length
			DC = DC/length """

			# Print the log info
			""" print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
					epoch+1, self.num_epochs, \
					epoch_loss,\
					acc,SE,SP,PC,F1,JS,DC))
 """
		

			# Decay learning rate
			""" if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
				lr -= (self.lr / float(self.num_epochs_decay))
				for param_group in self.optimizer.param_groups:
					param_group['lr'] = lr
				print ('Decay learning rate to lr: {}.'.format(lr)) """
		
					
			#===================================== Test ====================================#

	def test(self):
		""" del self.unet
		self.build_model()
		self.unet.load_state_dict(tf.io.read_file(unet_path))
		
		self.unet.train(False) """
		#self.unet.eval()

		acc = 0.	# Accuracy
		SE = 0.		# Sensitivity (Recall)
		SP = 0.		# Specificity
		PC = 0. 	# Precision
		F1 = 0.		# F1 Score
		JS = 0.		# Jaccard Similarity
		DC = 0.		# Dice Coefficient
		length=0
		for i in range(0,len(self.test_inputs),self.batch_size):
			# GT : Ground Truth

			images = self.test_inputs[i:i+self.batch_size]
			#print('Images:  ', images)
			GT = self.test_labels[i: i+self.batch_size]

			SR = activations.softmax(self.unet(images))

			#Displaying Images
			SR_image = tensor_to_image(SR[0])
			GT_image = tensor_to_image(GT[0])
			SR_image.show()
			GT_image.show()

			
			acc += get_accuracy(SR,GT)
			""" SE += get_sensitivity(SR,GT)
			SP += get_specificity(SR,GT)
			PC += get_precision(SR,GT)
			F1 += get_F1(SR,GT)
			JS += get_JS(SR,GT)
			DC += get_DC(SR,GT) """
					
			length += 1
				
		acc = acc/length
		print(acc)
		""" SE = SE/length
		SP = SP/length
		PC = PC/length
		F1 = F1/length
		JS = JS/length
		DC = DC/length """
		#unet_score = JS + DC


		#f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
		#wr = csv.writer(f)
		#wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC,self.lr,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
		#f.close()