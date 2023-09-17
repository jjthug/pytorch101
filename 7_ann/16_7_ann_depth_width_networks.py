import numpy as np
import torch
import torch.nn as nn

# build 2 models

widenet = nn.Sequential(
    nn.Linear(2,4),
    nn.Linear(4,3)
)

deepnet = nn.Sequential(
    nn.Linear(2,2),
    nn.Linear(2,2),
    nn.Linear(2,3)
)


print(widenet)
print(' ')
print(deepnet)

for p in deepnet.named_parameters():
    print(p)



numNodesinWide = 0

for p in widenet.named_parameters():
    if 'bias' in p[0]:
        numNodesinWide += len(p[1])

numNodesinDeep = 0

for p in deepnet.named_parameters():
    if 'bias' in p[0]:
        numNodesinDeep += len(p[1])

print("numNodesinWide=",numNodesinWide)
print("numNodesinDeep=",numNodesinDeep)

# print just the parameters

nparams = 0

for p in widenet.parameters():
    print(p)
    print(' ')
    if p.requires_grad:
        nparams += p.numel()

print("nparams=",nparams)

nparams = np.sum([p.numel() for p in widenet.parameters() if p.requires_grad])
print("widenet nparams=",nparams)

nparams = np.sum([p.numel() for p in deepnet.parameters() if p.requires_grad])
print("deepnet nparams=",nparams)


# from torchsummary import summary
#
# summary(widenet,(1,2))




