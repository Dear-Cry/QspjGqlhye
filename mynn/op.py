from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        W, b = self.params['W'], self.params['b']
        output = np.dot(X, W) + b
        return output

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        W = self.params['W']
        X = self.input
        self.grads['W'] = np.dot(X.T, grad)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        output = np.dot(grad, W.T)
        return output
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(out_channels, 1))
        self.grads = {'W': None, 'b': None}
        self.params = {'W': self.W, 'b': self.b}
        self.input = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        self.input = X
        batch_size, in_channels, H, W = X.shape
        out_channels = self.out_channels
        kH, kW = self.kernel_size

        new_H = (H - kH + 2 * self.padding) // self.stride + 1
        new_W = (W - kW + 2 * self.padding) // self.stride + 1

        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            X_padded = X

        output = np.zeros((batch_size, out_channels, new_H, new_W))

        # for b in range(batch_size):
        #     for c_out in range(out_channels):
        #         for h in range(new_H):
        #             for w in range(new_W):
        #                 h_start = h * self.stride
        #                 w_start = w * self.stride

        #                 receptive_field = X_padded[b, :, h_start:h_start+kH, w_start:w_start+kW]
        #                 output[b, c_out, h, w] = np.sum(receptive_field * self.W[c_out]) + self.b[c_out]
        for h in range(new_H):
            for w in range(new_W):
                h_start = h * self.stride
                w_start = w * self.stride
                # 一次性获取所有样本和通道的局部区域 [batch, in_ch, kH, kW]
                receptive_field = X_padded[:, :, h_start:h_start+kH, w_start:w_start+kW]
                # 向量化计算 [batch, out_ch, in_ch, kH, kW] * [out_ch, in_ch, kH, kW] -> [batch, out_ch]
                output[:, :, h, w] = np.tensordot(receptive_field, self.W, axes=([1,2,3], [1,2,3])) + self.b.T
        return output

    
    def backward(self, grads):
        X = self.input
        batch_size, in_channels, H, W = X.shape
        out_channels = self.out_channels
        kH, kW = self.kernel_size

        # 初始化梯度
        dW = np.zeros_like(self.W)  # shape [out_ch, in_ch, kH, kW]
        db = np.zeros_like(self.b)   # shape [out_ch, 1]
        dX = np.zeros_like(X)       # shape [batch, in_ch, H, W]

        # 处理padding
        if self.padding > 0:
            X_padded = np.pad(X, ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)))
            dX_padded = np.pad(dX, ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)))
        else:
            X_padded = X
            dX_padded = dX

        # 计算输出尺寸
        out_H = (H - kH + 2*self.padding) // self.stride + 1
        out_W = (W - kW + 2*self.padding) // self.stride + 1

        # 向量化计算
        for h in range(out_H):
            for w in range(out_W):
                h_start = h * self.stride
                w_start = w * self.stride
                
                # 获取感受野区域 [batch, in_ch, kH, kW]
                receptive_field = X_padded[:, :, h_start:h_start+kH, w_start:w_start+kW]
                
                # 计算dW [out_ch, in_ch, kH, kW]
                dW += np.einsum('bo,bijk->oijk', grads[:, :, h, w], receptive_field)
                
                # 计算dX [batch, in_ch, kH, kW]
                dX_padded[:, :, h_start:h_start+kH, w_start:w_start+kW] += \
                    np.einsum('bo,oijk->bijk', grads[:, :, h, w], self.W)
        
        # 计算db
        db = np.sum(grads, axis=(0,2,3)).reshape(-1, 1)
        
        # 移除padding
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded
            
        self.grads['W'] = dW / batch_size
        self.grads['b'] = db / batch_size
        
        return dX  # shape [batch, in_ch, H, W]
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output
    
class Logistic(Layer):
    pass

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.input = None
        self.labels = None
        self.grads = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        self.input = predicts
        self.labels = labels
        if self.has_softmax:
            probs = softmax(predicts)
        else:
            probs = predicts
        N = predicts.shape[0]
        log_probs = -np.log(probs[np.arange(N), labels] + 1e-10)
        loss = np.sum(log_probs) / N

        dloss = probs.copy()
        dloss[np.arange(N), labels] -= 1
        self.grads = dloss / N

        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, model=None, reg=1e-4) -> None:
        super().__init__()
        self.model = model
        self.reg = reg
        self.optimizable = False

    def forward(self, params):
        l2_loss = 0.
        for key in params:
            if key in ['W', 'b']:
                l2_loss += 0.5 * self.reg * np.sum(params[key] ** 2)
        return l2_loss
    
    def backward(self, params):
        grads = {}
        for key in params:
            if key in ['W', 'b']:
                grads[key] = self.reg * params[key]
        return grads
    
class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None
        self.optimizable = False  

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1)  

    def backward(self, grad):
        return grad.reshape(self.input_shape)  
    
class Dropout(Layer):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p
        self.mask = None
        self.optimizable = False
    
    def __call__(self, X, training=True):
        return self.forward(X, training)
    
    def forward(self, X, training=True):
        if training:
            self.mask = (np.random.rand(*X.shape) < self.p) / self.p
            return X * self.mask
        else:
            return X
    
    def backward(self, grad):
        return grad * self.mask 
    
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition