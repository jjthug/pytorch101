## Build model
import torch
from torch import nn
import matplotlib.pyplot as plt

class LinearRegressionModel(nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.weights=nn.Parameter(torch.randn((1),requires_grad=True,dtype=torch.float))
    self.bias=nn.Parameter(torch.randn((1),requires_grad=True,dtype=torch.float))

  def forward(self,x:torch.Tensor):
    return self.weights * x + self.bias

torch.manual_seed(23)

model_0 = LinearRegressionModel()

print(list(model_0.parameters()))

print(model_0.state_dict())


## create train/test data
weight = 0.7
bias = 0.3
start=0
end=1
step=0.02

x = torch.arange(start,end,step).unsqueeze(1)
y=weight*x + bias

split_test = int(0.8*len(x))
x_train, y_train = x[:split_test], y[:split_test]
x_test, y_test = x[split_test:], y[split_test:]

def plot_predictions(
    train_data=x_train,
    train_labels=y_train,
    test_data=x_test,
    test_labels=y_test,
    predictions=None
):

  plt.figure(figsize=(10,7))

  plt.scatter(train_data, train_labels,c="g",s=4,label="Training data")

  plt.scatter(test_data,test_labels,s=4,c="b",label="Test data")

  if predictions is not None:
    plt.scatter(test_data,predictions,c="r",s=4,label="Predictions")

  plt.legend(prop={"size":14})

print("x_test= ",x_test)


with torch.inference_mode():
  y_preds = model_0(x_test)

print(y_preds)

plot_predictions(predictions=y_preds)

# Loss function
loss_fn = nn.L1Loss()


#Optimizer
