# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
# np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]

# train_imgs, train_labs = nn.augment.data_augment(train_imgs, train_labs, ['rotate',])

class_num = 10
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
classes = classes[: class_num]
validation_num = 10000
train_num = 6000 * class_num - validation_num
# valid_imgs = train_imgs[:10000]
# valid_labs = train_labs[:10000]
# train_imgs = train_imgs[10000:]
# train_labs = train_labs[10000:]

train_imgs, train_labs = nn.preprocess.filter_by_class(train_imgs, train_labs, class_num)
train_imgs, train_labs, valid_imgs, valid_labs = nn.preprocess.train_validation_split(train_imgs, train_labs, train_num, validation_num)

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 256, 64, 10], 'ReLU', [1e-1, 1e-1, 1e-1])
# optimizer = nn.optimizer.SGD(init_lr=0.005, model=linear_model)
# optimizer = nn.optimizer.MomentGD(init_lr=0.005, model=linear_model)
optimizer = nn.optimizer.Adam(init_lr=0.001, model=linear_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, batch_size=64, scheduler=scheduler)
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=500 , log_iters=50, save_dir=r'./best_models')

# cnn_model = nn.models.Model_CNN(
#         conv_config=[(1, 8, 3, 1, 0, True, 1e-4), # in_channels, out_channels, kernel_size, stride, padding, weight_decay, weight_decay_lambda
#                      (8, 4, 5, 3, 0, True, 1e-4)], 
#         linear_config=[[10], (True, 1e-4)], # layer_dims, weight_decay
#         act_func='ReLU') 
# # optimizer = nn.optimizer.SGD(init_lr=1e-3, model=cnn_model)
# # optimizer = nn.optimizer.MomentGD(init_lr=0.005, model=cnn_model)
# optimizer = nn.optimizer.Adam(init_lr=0.005, model=cnn_model)
# scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[], gamma=0.5)
# loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max()+1)

# runner = nn.runner.RunnerC(cnn_model, optimizer, nn.metric.accuracy, loss_fn, batch_size=64, scheduler=scheduler)
# runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=200 , log_iters=10, save_dir=r'./best_models')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.savefig("fig.png")
plt.show()