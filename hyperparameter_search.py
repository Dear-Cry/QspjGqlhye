# you may do your own hyperparameter search job here.
import numpy as np
import random
import itertools
import pickle
import gzip
from struct import unpack
import mynn as nn
import matplotlib.pyplot as plt

print("Loading MNIST dataset...")
with gzip.open('./dataset/MNIST/train-images-idx3-ubyte.gz', 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, -1) 
with gzip.open('./dataset/MNIST/train-labels-idx1-ubyte.gz', 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

idx = np.random.permutation(np.arange(num))
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]

class_num = 10
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
classes = classes[: class_num]
validation_num = 5000
train_num = 3000 * class_num - validation_num

train_imgs, train_labs = nn.preprocess.filter_by_class(train_imgs, train_labs, class_num)
train_imgs, train_labs, valid_imgs, valid_labs = nn.preprocess.train_validation_split(train_imgs, train_labs, train_num, validation_num)

train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

param_grid = {
    'lr': [0.01, 0.005, 0.001],
    'h1': [256],
    'h2': [64],
    'reg': [1e-1, 1e-2, 1e-3],
    'batch_size': [32, 64, 128]
}

all_combinations = [dict(zip(param_grid.keys(), values)) 
                    for values in itertools.product(*param_grid.values())]

P = train_imgs.shape[1]
C = class_num
results = []

for params in all_combinations:
    print(f"\nTraining with params: {params}")
    
    h1, h2 = params['h1'], params['h2']
    layer_dims = [P, h1, h2, C]
    regs = [params['reg']] * 3 
    init_lr = params['lr']
    
    linear_model = nn.models.Model_MLP(layer_dims, 'ReLU', regs)
    optimizer = nn.optimizer.Adam(init_lr=init_lr, model=linear_model)
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[], gamma=0.5)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

    runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, batch_size=64, scheduler=scheduler)
    runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=500 , log_iters=50, save_dir=r'./best_models')
    
    train_acc = runner.train_scores[-1]
    val_acc = runner.dev_scores[-1]
    
    results.append({
        'params': params,
        'train_acc': train_acc,
        'val_acc': val_acc
    })

param_labels = [f"lr={p['lr']}\nh1={p['h1']}\nh2={p['h2']}\nreg={p['reg']}\nbatch_size={p['batch_size']}" 
                for p in [res['params'] for res in results]]
train_accs = [res['train_acc'] for res in results]
val_accs = [res['val_acc'] for res in results]

plt.figure(figsize=(20, 8))
x = np.arange(len(param_labels))
plt.bar(x - 0.2, train_accs, width=0.4, label='Training Accuracy')
plt.bar(x + 0.2, val_accs, width=0.4, label='Validation Accuracy')
plt.xticks(x, param_labels, rotation=45, ha='right')
plt.ylabel('Accuracy')
plt.title('Hyperparameter Comparison')
plt.legend()
plt.tight_layout()
plt.savefig("hyperparameter_comparison.png")
plt.show()

best_result = max(results, key=lambda x: x['val_acc'])
print(f"\nBest Params: {best_result['params']}, Val Acc: {best_result['val_acc']:.4f}")