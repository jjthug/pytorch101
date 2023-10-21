import torch
import torch.nn as nn
import torch.nn.functional as F


prob = .5
dropout = nn.Dropout(p=prob)
x=torch.ones(10)

y = dropout(x)

print(x)
print(y)
print(torch.mean(y))



# dropout turned off
dropout.eval()
y = dropout(x)

print(x)
print(y)
print(torch.mean(y))


dropout.eval()
y = F.dropout(x)
print(y)
print(torch.mean(y))


# but you can manually switch it off
# dropout.eval()
y = F.dropout(x,training=False)

print(y)
print(torch.mean(y))


# y = F.dropout(x,training=self.training)
# print(y)
# print(torch.mean(y))


# the model needs to be reset after toggling into eval mode

dropout.train()
y = dropout(x)
print(y) # with dropout


dropout.eval()
y = dropout(x)
print(y) # without dropout


# dropout.train()
y = dropout(x)
print(y) # still w/o dropout ;)