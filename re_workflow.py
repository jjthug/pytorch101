import torch.cuda
import matplotlib.pyplot as plt
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"device=>{device}")

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X + bias

split = int(0.8 * len(y))
x_train , y_train = X[0:split], y[0:split]
x_test, y_test = X[split:], y[split:]

def plot_predictions(
        train_data = x_train,
        train_labels = y_train,
        test_data = x_test,
        test_labels = y_test,
        predictions = None
):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_labels,c="g",s=4,label="Training data")
    plt.scatter(test_data,test_labels,c="b",s=4,label="Test data")
    if predictions is not None:
        plt.scatter(test_data,predictions,c="r",s=4,label="Predicitions")

    plt.legend(prop={"size":14})


class LinearRegressionModelV2(nn.Module):
    def __init__(self, *args, **kwargs):
        # Use
        super().__init__(*args, **kwargs)
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

torch.manual_seed(69)
model = LinearRegressionModelV2()
print("state dict of model =>",model.state_dict())

print(next((model.parameters())).device)
model.to(device)
print(next(model.parameters()).device)



## Set up loss functions

optimizer = torch.optim.SGD(params=model.parameters(),lr=0.001)
loss_fn = nn.L1Loss()

torch.manual_seed(69)

epochs =200

x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    model.train()
    y_pred = model(x_train)
    loss = loss_fn(y_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_pred = model(x_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss => {loss} | Test_loss => {test_loss}")

print(model.state_dict())

## Saving the models

from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)

MODEL_NAME= "model_1.pth"
MODEL_SAVE_APTH = MODEL_PATH / MODEL_NAME
1
print(f"model path => {MODEL_SAVE_APTH}")

torch.save(model.state_dict(),f=MODEL_SAVE_APTH)

loaded_model = LinearRegressionModelV2()
loaded_model.load_state_dict(torch.load(MODEL_SAVE_APTH))
loaded_model.to(device)
print(f"loaded model state => {loaded_model.state_dict()}")

# evaluate the loaded model

model.eval()
with torch.inference_mode():
    l_model_pred = model(x_test)
    l_loss = loss_fn(l_model_pred,y_test)
    print(f"l_loss => {l_loss}")
