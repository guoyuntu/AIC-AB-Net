# Image Captioning On General Data And Fashion Data

**This is a PyTorch implementation of Image Captioning On General Data And Fashion Data**

# Requirement

To train the model, you need a GPU and the Python 3.7 environment with the following pakages,
- Torch
- Torchvision
- NLTK

# Dataset

The Fashion dataset is not published but you are free the train and test the model on [MSCOCO]("https://cocodataset.org/#home").

You might want to use the following command to download the MSCOCO dataset:

	wget http://images.cocodataset.org/zips/train2014.zip
	wget http://images.cocodataset.org/zips/val2014.zip

Andrej Karpathy's training, validation, and test splits are used in this project. Use the following command to download the file:

	wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

Then put the data on the 'data' folder.

# Data Preprocessing

Run the following command to preprocessing the MSCOCO data.

	python resized.py

# Training

Use the following command to train the model on MSCOCO dataset.

	python train.py
