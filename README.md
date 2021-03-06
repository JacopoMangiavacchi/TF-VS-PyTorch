# Dynamic Subclassed Models with AutoDiff on PyTorch Vs TensorFlow

Comparing AutoDiff Dynamic Model approaches between PyTorch and TensorFlow 2.x  training from scratch a Linear Regression with a custom dynamic model class/module and manual training loop / loss function

## TensorFlow Dynamic Model
```Python
class LinearRegressionKeras(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(tf.random.uniform(shape=[1], minval=-0.1, maxval=0.1))
        self.b = tf.Variable(tf.random.uniform(shape=[1], minval=-0.1, maxval=0.1))

    def __call__(self,x): 
        return x * self.w + self.b
```

## PyTorch Dynamic Model
```Python
class LinearRegressionPyTorch(torch.nn.Module): 
    def __init__(self): 
        super().__init__()
        self.w = torch.nn.Parameter(torch.Tensor(1, 1).uniform_(-0.1, 0.1))
        self.b = torch.nn.Parameter(torch.Tensor(1).uniform_(-0.1, 0.1))
  
    def forward(self, x):  
        return x @ self.w + self.b
```

## TensorFlow Training Loop
```Python
def squared_error(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

tf_model = LinearRegressionKeras()
[w, b] = tf_model.trainable_variables

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        predictions = tf_model(x)
        loss = squared_error(predictions, y)
        
    w_grad, b_grad = tape.gradient(loss, tf_model.trainable_variables)

    w.assign(w - w_grad * learning_rate)
    b.assign(b - b_grad * learning_rate)

    if epoch % 20 == 0:
        print(f"Epoch {epoch} : Loss {loss.numpy()}")
```

## PyTorch Training Loop
```Python
def squared_error(y_pred, y_true):
    return torch.mean(torch.square(y_pred - y_true))

torch_model = LinearRegressionPyTorch()
[w, b] = torch_model.parameters()

for epoch in range(epochs):
    y_pred = torch_model(inputs)
    loss = squared_error(y_pred, labels)

    loss.backward()

    with torch.no_grad():
        w -= w.grad * learning_rate
        b -= b.grad * learning_rate
        w.grad.zero_()
        b.grad.zero_()

    if epoch % 20 == 0:
      print(f"Epoch {epoch} : Loss {loss.data}")
```

## TensorFlow Dynamic Model with Linear Layer
```Python
class LinearRegressionKeras(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.linear = tf.keras.layers.Dense(1, activation=None) # , input_shape=[1]

    def call(self, x): 
        return self.linear(x)
```

## PyTorch Dynamic Model with Linear Layer
```Python
class LinearRegressionPyTorch(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionPyTorch, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  

    def forward(self, x):
        return self.linear(x)
```

## TensorFlow Training Loop with Fit
```Python
tf_model_fit = LinearRegressionKeras()
tf_model_fit.compile(optimizer=tf.optimizers.SGD(learning_rate=learning_rate), loss='mean_absolute_error')
# tf_model_fit.build(input_shape=(None, 1))
# tf_model_fit.summary()
tf_model_fit.fit(x, y, epochs=epochs, batch_size=x.shape[0], verbose=1)
```

## TensorFlow Training Loop with real Loss function and Optimizer
```Python
tf_model_train_loop = LinearRegressionKeras()

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

for epoch in range(epochs * 3):
    x_batch = tf.reshape(x, [200, 1])
    with tf.GradientTape() as tape:
        y_pred = tf_model_train_loop(x_batch)
        y_pred = tf.reshape(y_pred, [200])
        loss = tf.losses.mse(y_pred, y)
    
    grads = tape.gradient(loss, tf_model_train_loop.variables)
    
    optimizer.apply_gradients(grads_and_vars=zip(grads, tf_model_train_loop.variables))

    if epoch % 20 == 0:
        print(f"Epoch {epoch} : Loss {loss.numpy()}")
```

## PyTorch Training Loop with real Loss function and Optimizer
```Python
torch_model = LinearRegressionPyTorch()

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)

for epoch in range(epochs * 3):
    y_pred = torch_model(inputs)
    loss = criterion(y_pred, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
      print(f"Epoch {epoch} : Loss {loss.data}")
```
