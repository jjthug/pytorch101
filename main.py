import torch as torch

weight = 0.7
bias = 0.3
start=0
end=1
step=0.02

x = torch.arange(start,end,step).unsqueeze(1)
y=weight*x + bias
print(x[:10],y[:10])


split_test = int(0.8*len(x))
x_train, y_train = x[:split_test], y[:split_test]
x_test, y_test = x[split_test:], y[split_test:]

import matplotlib.pyplot as plt
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

plot_predictions()