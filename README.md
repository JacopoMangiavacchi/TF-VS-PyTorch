# TF-VS-PyTorch Dynamic Model

Comparing AutoGrad Dynamic Model approaches between PyTorch and TensorFlow 2.x  training from scratch a Linear Regression with a custom dynamic model class/module and manual training loop / loss function

## TensorFlow Dynamic Model
```Python
class LinearRegressionKeras(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(tf.random.uniform(shape=[1], minval=-0.1, maxval=0.1))
        self.b = tf.Variable(tf.random.uniform(shape=[1], minval=-0.1, maxval=0.1))

    def __call__(self,x): 
        y_pred = x * self.w + self.b
        return y_pred
```

## PyTorch Dynamic Model
```Python
class LinearRegressionPyTorch(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.w = torch.nn.Parameter(torch.Tensor(1, 1))
        self.b = torch.nn.Parameter(torch.Tensor(1))
        self.w.data.uniform_(-0.1, 0.1)
        self.b.data.uniform_(-0.1, 0.1)
  
    def forward(self, x):  
        return x @ self.w + self.b
```


