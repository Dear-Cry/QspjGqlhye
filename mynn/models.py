from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout_p=1.0):
        self.size_list = size_list
        self.act_func = act_func
        self.dropout_p = dropout_p
        self.mode = 'train' # or 'test'

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                if dropout_p < 1.0:
                    layer_f = Dropout(dropout_p)
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)
    
    def set_mode(self, mode):
        self.mode = mode

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            if isinstance(layer, Dropout):
                if self.mode == 'train':
                    outputs = layer(outputs)
                else:
                    outputs = layer(outputs, False)
            else:
                outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i + 2]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplemented
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, conv_config=None, linear_config=None, act_func='ReLU'):
        super().__init__()
        self.conv_config = conv_config
        self.linear_config = linear_config
        self.act_func = act_func
        self.layers = []
        if conv_config is not None:
            for i, (in_ch, out_ch, k_size, stride, padding, weight_decay, weight_decay_lambda) in enumerate(conv_config):
                self.layers.append(conv2D(in_channels=in_ch, 
                                         out_channels=out_ch, 
                                         kernel_size=k_size, 
                                         stride=stride,
                                         padding=padding,
                                         weight_decay=weight_decay,
                                         weight_decay_lambda=weight_decay_lambda))
                if i < len(conv_config) - 1 or linear_config is not None:  
                    if act_func == 'ReLU':
                        self.layers.append(ReLU())
                    elif act_func == 'Logistic':
                        raise NotImplementedError
        
        if conv_config:
            self.layers.append(Flatten()) 

        if linear_config is not None:
            if conv_config:
                h, w = 28, 28
                for _, _, k_size, stride, padding, _, _ in conv_config:
                    h = (h - k_size + 2 * padding) // stride + 1  
                    w = (w - k_size + 2 * padding) // stride + 1
                flattened_size = conv_config[-1][1] * h * w  # channels * H * W
            else:
                flattened_size = 28 * 28 

            full_linear_config = [flattened_size] + linear_config[0]
            weight_decay, weight_decay_lambda = linear_config[1]
            
            for i in range(len(full_linear_config) - 1):
                self.layers.append(Linear(in_dim=full_linear_config[i], 
                                         out_dim=full_linear_config[i+1],
                                         weight_decay=weight_decay,
                                         weight_decay_lambda=weight_decay_lambda))
                if i < len(full_linear_config) - 2: 
                    if act_func == 'ReLU':
                        self.layers.append(ReLU())
                    elif act_func == 'Logistic':
                        raise NotImplementedError

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        outputs = X
        if len(outputs.shape) == 2:
            outputs = outputs.reshape(-1, 1, 28, 28)
            
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        
        self.act_func = param_list[0]
        self.layers = []

        for layer_params in param_list[1:]:
            if layer_params['type'] == 'conv2D':
                layer = conv2D(
                    in_channels=layer_params['in_channels'],
                    out_channels=layer_params['out_channels'],
                    kernel_size=layer_params['kernel_size'],
                    stride=layer_params['stride'],
                    padding=layer_params['padding'], 
                    weight_decay=layer_params['weight_decay'],
                    weight_decay_lambda=layer_params['lambda']
                )
                layer.W = layer_params['W']
                layer.b = layer_params['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
            elif layer_params['type'] == 'Linear':
                layer = Linear(
                    in_dim=layer_params['in_dim'],
                    out_dim=layer_params['out_dim'],
                    weight_decay=layer_params['weight_decay'],
                    weight_decay_lambda=layer_params['lambda']
                )
                layer.W = layer_params['W']
                layer.b = layer_params['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
            elif layer_params['type'] == 'ReLU':
                layer = ReLU()
            elif layer_params['type'] == 'Flatten':
                layer = Flatten()
            else:
                raise ValueError(f"Unsupported layer type: {layer_params['type']}")
            self.layers.append(layer)

        
    def save_model(self, save_path):
        param_list = [self.act_func]
        for layer in self.layers:
            if isinstance(layer, conv2D):
                layer_params = {
                    'type': 'conv2D',
                    'in_channels': layer.in_channels,
                    'out_channels': layer.out_channels,
                    'kernel_size': layer.kernel_size[0],
                    'stride': layer.stride,
                    'padding': layer.padding,
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                }
            elif isinstance(layer, Linear):
                layer_params = {
                    'type': 'Linear',
                    'in_dim': layer.params['W'].shape[0],
                    'out_dim': layer.params['W'].shape[1],
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                }
            elif isinstance(layer, ReLU):
                layer_params = {'type': 'ReLU'}
            elif isinstance(layer, Logistic):
                layer_params = {'type': 'Logistic'}
            elif isinstance(layer, Flatten):  
                layer_params = {'type': 'Flatten'}
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")
            
            param_list.append(layer_params)
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)