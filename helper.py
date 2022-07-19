
import os
import shutil


def progress_percentage(perc, width=None):
	# This will only work for python 3.3+ due to use of
	# os.get_terminal_size the print function etc.

	FULL_BLOCK = '█'
	# this is a gradient of incompleteness
	INCOMPLETE_BLOCK_GRAD = ['░', '▒', '▓']

	assert(isinstance(perc, float))
	assert(0. <= perc <= 100.)
	# if width unset use full terminal
	if width is None:
		width = os.get_terminal_size().columns
	# progress bar is block_widget separator perc_widget : ####### 30%
	max_perc_widget = '[100.00%]' # 100% is max
	separator = ' '
	blocks_widget_width = width - len(separator) - len(max_perc_widget)
	assert(blocks_widget_width >= 10) # not very meaningful if not
	perc_per_block = 100.0/blocks_widget_width
	# epsilon is the sensitivity of rendering a gradient block
	epsilon = 1e-6
	# number of blocks that should be represented as complete
	full_blocks = int((perc + epsilon)/perc_per_block)
	# the rest are "incomplete"
	empty_blocks = blocks_widget_width - full_blocks

	# build blocks widget
	blocks_widget = ([FULL_BLOCK] * full_blocks)
	blocks_widget.extend([INCOMPLETE_BLOCK_GRAD[0]] * empty_blocks)
	# marginal case - remainder due to how granular our blocks are
	remainder = perc - full_blocks*perc_per_block
	# epsilon needed for rounding errors (check would be != 0.)
	# based on reminder modify first empty block shading
	# depending on remainder
	if remainder > epsilon:
		grad_index = int((len(INCOMPLETE_BLOCK_GRAD) * remainder)/perc_per_block)
		blocks_widget[full_blocks] = INCOMPLETE_BLOCK_GRAD[grad_index]

	# build perc widget
	str_perc = '%.2f' % perc
	# -1 because the percentage sign is not included
	perc_widget = '[%s%%]' % str_perc.ljust(len(max_perc_widget) - 3)

	# form progressbar
	progress_bar = '%s%s%s' % (''.join(blocks_widget), separator, perc_widget)
	# return progressbar as string
	return ''.join(progress_bar)


def copy_progress(copied, total):
	print('\r' + progress_percentage(100*copied/total, width=30), end='')


def copyfile(src, dst, *, follow_symlinks=True):
	"""Copy data from src to dst.

	If follow_symlinks is not set and src is a symbolic link, a new
	symlink will be created instead of copying the file it points to.

	"""
	if shutil._samefile(src, dst):
		raise shutil.SameFileError("{!r} and {!r} are the same file".format(src, dst))

	for fn in [src, dst]:
		try:
			st = os.stat(fn)
		except OSError:
			# File most likely does not exist
			pass
		else:
			# XXX What about other special files? (sockets, devices...)
			if shutil.stat.S_ISFIFO(st.st_mode):
				raise shutil.SpecialFileError("`%s` is a named pipe" % fn)

	if not follow_symlinks and os.path.islink(src):
		os.symlink(os.readlink(src), dst)
	else:
		size = os.stat(src).st_size
		with open(src, 'rb') as fsrc:
			with open(dst, 'wb') as fdst:
				copyfileobj(fsrc, fdst, callback=copy_progress, total=size)
	return dst


def copyfileobj(fsrc, fdst, callback, total, length=16*1024):
	copied = 0
	while True:
		buf = fsrc.read(length)
		if not buf:
			break
		fdst.write(buf)
		copied += len(buf)
		callback(copied, total=total)


def copy_with_progress(src, dst, *, follow_symlinks=True):
	if os.path.isdir(dst):
		dst = os.path.join(dst, os.path.basename(src))
	copyfile(src, dst, follow_symlinks=follow_symlinks)
	shutil.copymode(src, dst)
	return dst

import numpy as np
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator

from constants import *

def cnn_model():
	
	model = Sequential()
	
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS)))    
	model.add(Conv2D(32, (3, 3), activation='relu'))    
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))    
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	
	model.add(Dense(num_classes, activation='softmax'))
	
	model.summary()
	
	return model
	
def pure_cnn_model():
	
	model = Sequential()
	
	model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same', input_shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS)))    
	model.add(Dropout(0.2))
	
	model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same'))  
	model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2))    
	model.add(Dropout(0.5))
	
	model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same'))    
	model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same'))
	model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2))    
	model.add(Dropout(0.5))    
	
	model.add(Conv2D(192, (3, 3), padding = 'same'))
	model.add(Activation('relu'))
	model.add(Conv2D(192, (1, 1),padding='valid'))
	model.add(Activation('relu'))
	model.add(Conv2D(10, (1, 1), padding='valid'))

	model.add(GlobalAveragePooling2D())
	
	model.add(Activation('softmax'))

	model.summary()
	
	return model


# CIFAR - 10

# To decode the files
import pickle
# For array manipulations
import numpy as np
# To make one-hot vectors
from keras.utils import np_utils
# To plot graphs and display images
from matplotlib import pyplot as plt


#constants

path = "data/"  # Path to data 

# Height or width of the images (32 x 32)
size = 32 

# 3 channels: Red, Green, Blue (RGB)
channels = 3  

# Number of classes
num_classes = 10 

# Each file contains 10000 images
image_batch = 10000 

# 5 training files
num_files_train = 5  

# Total number of training images
images_train = image_batch * num_files_train

# https://www.cs.toronto.edu/~kriz/cifar.html


def unpickle(file):  
	
	# Convert byte stream to object
	with open(path + file,'rb') as fo:
		print("Decoding file: %s" % (path+file))
		dict = pickle.load(fo, encoding='bytes')
	   
	# Dictionary with images and labels
	return dict




def convert_images(raw_images):
	
	# Convert images to numpy arrays
	
	# Convert raw images to numpy array and normalize it
	raw = np.array(raw_images, dtype = float) / 255.0
	
	# Reshape to 4-dimensions - [image_number, channel, height, width]
	images = raw.reshape([-1, channels, size, size])

	images = images.transpose([0, 2, 3, 1])

	# 4D array - [image_number, height, width, channel]
	return images




def load_data(file):
	# Load file, unpickle it and return images with their labels
	
	data = unpickle(file)
	
	# Get raw images
	images_array = data[b'data']
	
	# Convert image
	images = convert_images(images_array)
	# Convert class number to numpy array
	labels = np.array(data[b'labels'])
		
	# Images and labels in np array form
	return images, labels




def get_test_data():
	# Load all test data
	
	images, labels = load_data(file = "test_batch")
	
	# Images, their labels and 
	# corresponding one-hot vectors in form of np arrays
	return images, labels, np_utils.to_categorical(labels,num_classes)




def get_train_data():
	# Load all training data in 5 files
	
	# Pre-allocate arrays
	images = np.zeros(shape = [images_train, size, size, channels], dtype = float)
	labels = np.zeros(shape=[images_train],dtype = int)
	
	# Starting index of training dataset
	start = 0
	
	# For all 5 files
	for i in range(num_files_train):
		
		# Load images and labels
		images_batch, labels_batch = load_data(file = "data_batch_" + str(i+1))
		
		# Calculate end index for current batch
		end = start + image_batch
		
		# Store data to corresponding arrays
		images[start:end,:] = images_batch        
		labels[start:end] = labels_batch
		
		# Update starting index of next batch
		start = end
	
	# Images, their labels and 
	# corresponding one-hot vectors in form of np arrays
	return images, labels, np_utils.to_categorical(labels,num_classes)


def plot_images(images, labels_true, class_names, labels_pred=None):

	assert len(images) == len(labels_true)

	# Create a figure with sub-plots
	fig, axes = plt.subplots(3, 3, figsize = (8,8))

	# Adjust the vertical spacing
	if labels_pred is None:
		hspace = 0.2
	else:
		hspace = 0.5
	fig.subplots_adjust(hspace=hspace, wspace=0.3)

	for i, ax in enumerate(axes.flat):
		# Fix crash when less than 9 images
		if i < len(images):
			# Plot the image
			ax.imshow(images[i], interpolation='spline16')
			
			# Name of the true class
			labels_true_name = class_names[labels_true[i]]

			# Show true and predicted classes
			if labels_pred is None:
				xlabel = "True: "+labels_true_name
			else:
				# Name of the predicted class
				labels_pred_name = class_names[labels_pred[i]]

				xlabel = "True: "+labels_true_name+"\nPredicted: "+ labels_pred_name

			# Show the class on the x-axis
			ax.set_xlabel(xlabel)
		
		# Remove ticks from the plot
		ax.set_xticks([])
		ax.set_yticks([])
	
	# Show the plot
	# plt.show()
	#plot_file_name = "plot_" + str(labels_true[0]) + ".png"
	#plt.savefig('.png')

	

def plot_model(model_details):

	# Create sub-plots
	fig, axs = plt.subplots(1,2,figsize=(15,5))
	
	# Summarize history for accuracy
	axs[0].plot(range(1,len(model_details.history['accuracy'])+1),model_details.history['accuracy'])
	axs[0].plot(range(1,len(model_details.history['val_accuracy'])+1),model_details.history['val_accuracy'])
	axs[0].set_title('Model Accuracy')
	axs[0].set_ylabel('Accuracy')
	axs[0].set_xlabel('Epoch')
	axs[0].set_xticks(np.arange(1,len(model_details.history['accuracy'])+1),len(model_details.history['accuracy'])/10)
	axs[0].legend(['train', 'val'], loc='best')
	
	# Summarize history for loss
	axs[1].plot(range(1,len(model_details.history['loss'])+1),model_details.history['loss'])
	axs[1].plot(range(1,len(model_details.history['val_loss'])+1),model_details.history['val_loss'])
	axs[1].set_title('Model Loss')
	axs[1].set_ylabel('Loss')
	axs[1].set_xlabel('Epoch')
	axs[1].set_xticks(np.arange(1,len(model_details.history['loss'])+1),len(model_details.history['loss'])/10)
	axs[1].legend(['train', 'val'], loc='best')
	
	# Show the plot
	#plt.show()



def visualize_errors(images_test, labels_test, class_names, labels_pred, correct):
	
	print("correct")
	print(correct)
	incorrect = (correct == False)
	#incorrect = np.logical_not(correct.equal(False))
	#incorrect = [not x for x in correct]
	
	# Images of the test-set that have been incorrectly classified.
	images_error = images_test[incorrect]
	
	# Get predicted classes for those images
	labels_error = labels_pred[incorrect]

	# Get true classes for those images
	labels_true = labels_test[incorrect]
	
	
	# Plot the first 9 images.
	plot_images(images=images_error[0:9],
				labels_true=labels_true[0:9],
				class_names=class_names,
				labels_pred=labels_error[0:9])
	
	
def predict_classes(model, images_test, labels_test):
	
	# Predict class of image using model
	class_pred = model.predict(images_test, batch_size=32)

	# Convert vector to a label
	labels_pred = np.argmax(class_pred,axis=1)

	# Boolean array that tell if predicted label is the true label
	correct = (labels_pred == labels_test)

	print(sum(correct))

	# Array which tells if the prediction is correct or not
	# And predicted labels
	return correct, labels_pred


