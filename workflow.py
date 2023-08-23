## Build model
import numpy as np
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
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)


epochs = 1000
epoch_arr = []
loss_arr = []
test_loss_arr = []

## Training

for epoch in range(epochs):

  model_0.eval()
  with torch.inference_mode():
    test_pred = model_0.forward(x_test)
    test_loss_arr.append(loss_fn(test_pred, y_test))

  epoch_arr.append(epoch)
  model_0.train()

  y_pred = model_0(x_train)

  loss = loss_fn(y_pred,y_train)

  optimizer.zero_grad()

  loss.backward()

  optimizer.step()

  model_0.eval()

  print(model_0.state_dict())
  print("loss=",loss)
  loss_arr.append(loss)

with torch.inference_mode():
  test_pred = model_0.forward(x_test)
  loss = loss_fn(test_pred,y_test)
  print("test loss =", loss)


# Plot the loss curves

plt.plot(epoch_arr, np.array(torch.tensor(loss_arr).numpy()), label="Training loss")
plt.plot(epoch_arr, np.array(torch.tensor(test_loss_arr).numpy()), label="Test loss")
plt.title("training and test loss curves")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()

## SAVING model

from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME="model_0.pth"
MODEL_SAVE_PATH=MODEL_PATH/MODEL_NAME

print(f"Saving model to {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)


## LOADING the model
print(model_0.state_dict())

loaded_model_0 = LinearRegressionModel()

loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

print("loaded model state dict => ",loaded_model_0.state_dict())

loaded_model_0.eval()

with torch.inference_mode():
  loaded_pred = loaded_model_0(x_test)

print("loaded pred =>",loaded_pred)

print("check if model and loaded model preds are equal =>",loaded_pred == y_preds)
