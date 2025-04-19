# codes to make visualization of your weights.
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

model = nn.models.Model_MLP()
model.load_model(r'.\saved_models\2HAdam.pickle')
W = model.layers[0].params['W']
W = W.T.reshape(-1, 28, 28)
n_cols = 16
n_plots = W.shape[0]
n_rows = n_plots // n_cols
plt.figure(figsize=(n_cols*2, n_rows*2))
for i in range(n_plots):
	plt.subplot(n_rows, n_cols, i+1)
	img = (W[i] - W[i].min()) / (W[i].max() - W[i].min())
	plt.imshow(img, cmap='gray')
	plt.axis('off')
plt.suptitle(f"Weight Visualization in the First Layer of MLP")
plt.savefig(f"W_image_MLP.png")

model = nn.models.Model_CNN()
model.load_model(r'.\saved_models\2HAdamCNN.pickle')
W_conv = model.layers[0].params['W']
W_conv = W_conv.squeeze(1)
out_channels, in_channels, kH, kW = model.layers[0].params['W'].shape
n_cols = int(np.sqrt(out_channels)) + 1
n_rows = int(np.ceil(out_channels / n_cols))
plt.figure(figsize=(n_cols*2, n_rows*2))
for i in range(out_channels):
	plt.subplot(n_rows, n_cols, i+1)
	img = W_conv[i]
	img = (img - img.min()) / (img.max() - img.min())
	plt.imshow(img, cmap='viridis')
	plt.axis('off')
plt.tight_layout()
plt.suptitle("Convolutional Kernels Visualization in the First Layer of CNN", y=1.02)
plt.savefig("CNN_kernels.png", bbox_inches='tight', dpi=300)
plt.show()
# mats = []
# mats.append(model.layers[0].params['W'])

# out_channels, in_channels, kH, kW = model.layers[0].params['W'].shape
# n_cols = int(np.sqrt(out_channels))
# n_rows = int(np.ceil(out_channels / n_cols))

# fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
# fig.suptitle('Conv Layer Weights')
# axes = axes.reshape(-1)
# for i in range(out_channels):
#         axes[i].matshow(mats[0][i].reshape(kH,kW))
#         axes[i].set_xticks([])
#         axes[i].set_yticks([])
# plt.tight_layout()
# plt.savefig('conv_weights.png', dpi=300, bbox_inches='tight')  

# plt.matshow(mats[1].T)
# plt.xticks([])
# plt.yticks([])
# plt.savefig('fc_weights.png', dpi=300, bbox_inches='tight')
