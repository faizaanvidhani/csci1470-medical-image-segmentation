import tensorflow as tf
from tensorflow.keras import optimizers
import tensorflow.keras.activations as activations
from evaluation import *
from network2 import R2U_Net
import matplotlib as mpl
mpl.use('tkagg')
from matplotlib import pyplot as plt


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
		self.model_type = 'R2U_Net'
		self.train_loss_list = []
		self.test_acc_list = []

	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
		
		# Train for Encoder
		
		# NOTE: self.num_epochs set to 1 for current implementation
		for epoch in range(self.num_epochs):

			epoch_loss = 0
			
			acc = 0.	# Accuracy

			for i in range(0,len(self.train_inputs),self.batch_size):
				# GT : Ground Truth

				images = self.train_inputs[i:i+self.batch_size]
				GT = self.train_labels[i: i+self.batch_size]

				# Backprop + optimize
				with tf.GradientTape() as tape:
					# SR : Segmentation Result
					SR = self.unet(images)
					SR_probs = activations.softmax(SR)
					SR_flat = tf.reshape(SR_probs, [tf.shape(SR_probs)[0],-1])
					GT_flat = tf.reshape(GT, [tf.shape(GT)[0], -1])
					bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
					loss = bce(GT_flat, SR_flat)
					avg_loss = tf.reduce_mean(loss)
					self.train_loss_list.append(avg_loss.numpy())
					print("loss is", avg_loss)
					epoch_loss += avg_loss

				gradients = tape.gradient(avg_loss, self.unet.trainable_variables)
				self.optimizer.apply_gradients(zip(gradients, self.unet.trainable_variables))	
			#===================================== Test ====================================#

	def test(self):
		acc = 0.	# Accuracy
		for i in range(0,len(self.test_inputs),self.batch_size):
			# GT : Ground Truth
			images = self.test_inputs[i:i+self.batch_size]
			GT = self.test_labels[i: i+self.batch_size]
			SR = self.unet(images)
			SR_probs = activations.softmax(SR)

			batch_acc = get_accuracy(SR_probs,GT)
			self.test_acc_list.append(batch_acc)
			acc += batch_acc
				
		acc = acc/len(self.test_acc_list)
		print("final acc is", acc)
		print("loss_list:", self.train_loss_list)
		print("acc_list:", self.test_acc_list)
	
	def visualize_loss(self):
		"""
		Uses Matplotlib to visualize the losses of our model.
		:param losses: list of loss data stored from train
		:return: doesn't return anything, a plot should pop-up 
		"""
		x = [i for i in range(len(self.train_loss_list))]
		plt.plot(x, self.train_loss_list)
		plt.title('TRAINING: Loss per batch')
		plt.xlabel('Batch')
		plt.ylabel('Loss')
		plt.show()
	
	def visualize_accuracy(self):
		"""
		Uses Matplotlib to visualize the batch accuracies of our model.
		:param losses: list of batch accuracy stored from test
		:return: doesn't return anything, a plot should pop-up 
		"""
		x = [i for i in range(len(self.test_acc_list))]
		plt.plot(x, self.test_acc_list)
		plt.title('TEST: Accuracy per batch')
		plt.xlabel('Batch')
		plt.ylabel('Accuracy')
		plt.show()    
