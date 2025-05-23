import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

model = nn.models.Model_MLP()
# model = nn.models.Model_CNN()
model.load_model(r'.\best_models\best_model.pickle')
# model.load_model(r'.\best_models\best_model.pickle')

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

class_num = 10
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
classes = classes[: class_num]

test_imgs, test_labs = nn.preprocess.filter_by_class(test_imgs, test_labs, class_num)
test_imgs = test_imgs / test_imgs.max()
logits = model(test_imgs)
print(nn.metric.accuracy(logits, test_labs))