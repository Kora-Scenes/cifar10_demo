import os
import glob

import numpy as np

from tqdm import tqdm
import cv2
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import sys
sys.path.append('../')

from helper import cnn_model
from pipeline_input import pipeline_data_visualizer, pipeline_dataset_interpreter, pipeline_ensembler, pipeline_model, pipeline_input
from constants import *

class cifar10_interp_1(pipeline_dataset_interpreter):
	def load(self) -> None:
		super().load()
		dataset = {}
		num_classes = 10
		for ti, t in ((0, "train"), (1, "test")):
			folder_path = os.path.join(self.input_dir, t)
			csv_file =  os.path.join(folder_path, "groundtruth.csv")
			images_len = len(glob.glob(os.path.join(folder_path, "*.png")))
			size = 32
			channels = 3
			dataset[t] = {
				'img': np.zeros(shape = [images_len, size, size, channels], dtype = float),
				'label': np.zeros(shape=[images_len],dtype = int),
				'class': []     # One hot encoded
			}
			f = open(csv_file, "r")
			for line_no, l in tqdm(enumerate(f.readlines()), total=images_len):
				img_file_path, label = l.split(",")
				#img_file_path = os.path.join(folder_path, img_file_path)
				img = cv2.imread(img_file_path)
				dataset[t]['img'][line_no] = img
				dataset[t]['label'][line_no] = int(label)
			dataset[t]['class'] = np_utils.to_categorical(dataset[t]['label'], num_classes)
			f.close()
		self.__dataset = dataset

class cifar10_pipeline_model_1(pipeline_model):

	def load(self):
		super().load()
		#model = pure_cnn_model()
		model = cnn_model()
		model.compile(loss='categorical_crossentropy', # Better loss function for neural networks
				optimizer=Adam(lr=LEARN_RATE), # Adam optimizer with 1.0e-4 learning rate
				metrics = ['accuracy']) # Metrics to be evaluated by the model
		self.__model = model
		
	def train(self, log_dir, dataset):
		checkpoint = ModelCheckpoint('best_model_improved.h5',  # model filename
								monitor='val_loss', # quantity to monitor
								verbose=0, # verbosity - 0 or 1
								save_best_only= True, # The latest best model will not be overwritten
								mode='auto') # The decision to overwrite model is made 
											# automatically depending on the quantity to monitor 

		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

		images_train = dataset['train']['img']
		class_train = dataset['train']['class']
		images_test = dataset['test']['img']
		class_test = dataset['test']['class']
		
		self.__model_details = self.__model.fit(images_train, class_train,
						batch_size = 128,
						epochs = 10, # number of iterations
						validation_data= (images_test, class_test),
						callbacks=[checkpoint, tensorboard_callback],
						verbose=1)
		#return self.__model, model_details

class cifar10_pipeline_ensembler_1(pipeline_ensembler):

	def merge(self, x: np.array) -> np.array:
		xm = np.zeros_like(x.shape[1:], dtype=x.dtype)
		for i in range(len(x.shape[0])):
			xm[i] = np.argmax(x[i])
		return xm

cifar10_input = pipeline_input("cifar10_cnn", {'cifar10_interp_1': cifar10_interp_1}, {'cifar10_pipeline_model_1': cifar10_pipeline_model_1}, {'cifar10_pipeline_ensembler_1': cifar10_pipeline_ensembler_1}, {})

# exported_pipeline = cifar10_input

#all_inputs = {}
#all_inputs[cifar10_input.get_pipeline_name()] = cifar10_input
