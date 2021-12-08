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
from network import R2U_Net
import csv


class Solver(object):
	def __init__(self, train_inputs, train_labels, test_inputs, test_labels):

		# Data loader
		self.train_inputs = train_inputs
		self.train_labels = train_labels
		self.test_inputs = test_inputs 
		self.test_labels = test_labels 

		# Models
		self.model = R2U_Net(self.img_ch,self.output_ch,t=self.t)
		self.optimizer = optimizers.Adam(self.lr, self.beta1, self.beta2)
		self.img_ch = 3
		self.output_ch = 1
		self.criterion = tf.keras.losses.BinaryCrossEntropy()
		self.augmentation_prob = 0.4

		# Hyper-parameters
		self.lr = 0.0002
		self.beta1 = 0.5
		self.beta2 = 0.999

		# Training settings
		self.num_epochs = 100
		self.num_epochs_decay = 70
		self.batch_size = 10

		# Step size
		self.log_step = 2
		self.val_step = 2

		self.device = tf.device('cpu')
		self.unet.to(self.device)
		self.model_type = 'R2U_Net'
		self.t = 3

""" 	def print_network(self, model, name):
		Print out the network information.
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params)) """

""" 	def to_data(self, x):
		#convert variable to tensor
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data """

""" 	def reset_grad(self):
		Zero the gradient buffers.
		self.unet.zero_grad() """

""" 	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)

		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)
		return acc """
	
	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img


	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
		
		# Train for Encoder
		lr = self.lr
		
		for epoch in range(self.num_epochs):

			self.unet.train(True)
			epoch_loss = 0
			
			acc = 0.	# Accuracy
			SE = 0.		# Sensitivity (Recall)
			SP = 0.		# Specificity
			PC = 0. 	# Precision
			F1 = 0.		# F1 Score
			JS = 0.		# Jaccard Similarity
			DC = 0.		# Dice Coefficient
			length = 0

			for i, (images, GT) in enumerate(self.train_loader):#no train loader
				# GT : Ground Truth

				images = images.to(self.device)
				GT = GT.to(self.device)

				# SR : Segmentation Result
				SR = self.unet(images)
				SR_probs = activations.sigmoid(SR)
				SR_flat = SR_probs.view(SR_probs.size(0),-1)

				GT_flat = GT.view(GT.size(0),-1)

				# Backprop + optimize
				""" self.reset_grad()
				loss.backward()
				self.optimizer.step() """
				with tf.GradientTape() as tape:
					loss = self.criterion(SR_flat,GT_flat)
					epoch_loss += loss.item()
				gradients = tape.gradient(loss, self.model.trainable_variables)
				self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

				acc += get_accuracy(SR,GT)
				SE += get_sensitivity(SR,GT)
				SP += get_specificity(SR,GT)
				PC += get_precision(SR,GT)
				F1 += get_F1(SR,GT)
				JS += get_JS(SR,GT)
				DC += get_DC(SR,GT)
				length += images.size(0)

			acc = acc/length
			SE = SE/length
			SP = SP/length
			PC = PC/length
			F1 = F1/length
			JS = JS/length
			DC = DC/length

			# Print the log info
			print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
					epoch+1, self.num_epochs, \
					epoch_loss,\
					acc,SE,SP,PC,F1,JS,DC))

		

			# Decay learning rate
			if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
				lr -= (self.lr / float(self.num_epochs_decay))
				for param_group in self.optimizer.param_groups:
					param_group['lr'] = lr
				print ('Decay learning rate to lr: {}.'.format(lr))
		
					
			#===================================== Test ====================================#

	def test(self):
		""" del self.unet
		self.build_model()
		self.unet.load_state_dict(tf.io.read_file(unet_path))
		
		self.unet.train(False) """
		self.unet.eval()

		acc = 0.	# Accuracy
		SE = 0.		# Sensitivity (Recall)
		SP = 0.		# Specificity
		PC = 0. 	# Precision
		F1 = 0.		# F1 Score
		JS = 0.		# Jaccard Similarity
		DC = 0.		# Dice Coefficient
		length=0
		for i, (images, GT) in enumerate(self.test_loader):

			images = images.to(self.device)
			GT = GT.to(self.device)
			SR = activations.sigmoid(self.unet(images))
			acc += get_accuracy(SR,GT)
			SE += get_sensitivity(SR,GT)
			SP += get_specificity(SR,GT)
			PC += get_precision(SR,GT)
			F1 += get_F1(SR,GT)
			JS += get_JS(SR,GT)
			DC += get_DC(SR,GT)
					
			length += images.size(0)
				
		acc = acc/length
		SE = SE/length
		SP = SP/length
		PC = PC/length
		F1 = F1/length
		JS = JS/length
		DC = DC/length
		#unet_score = JS + DC


		#f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
		#wr = csv.writer(f)
		#wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC,self.lr,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
		#f.close()