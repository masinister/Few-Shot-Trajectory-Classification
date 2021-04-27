import pickle
import numpy as np
import torch
from model import Siamese, SuperSiamese
from utils import split_traj, featurize
import random

validate = pickle.load(open("validate_set.pkl",'rb'))
label = pickle.load(open("validate_label.pkl",'rb'))

def load_model():
    model = SuperSiamese(17, 8).to(torch.device("cuda"))
    model.load_state_dict(torch.load('testmodel.pth'))
    model.eval()
    return model


def process_data(traj_1, traj_2):
    """
    Input:
      Traj: a list of list, contains one trajectory for one driver
      example:[[114.10437, 22.573433, '2016-07-02 00:08:45', 1],
         [114.179665, 22.558701, '2016-07-02 00:08:52', 1]]
    Output:
      Data: any format that can be consumed by your model.

    """
    data = []
    split1 = [featurize(sub, torch.device("cuda")) for sub in split_traj(traj_1, 64)]
    split2 = [featurize(sub, torch.device("cuda")) for sub in split_traj(traj_2, 64)]
    x1 = torch.stack([torch.stack(random.choices(split1, k = 8)) for _ in range(8)])
    x2 = torch.stack([torch.stack(random.choices(split2, k = 8)) for _ in range(8)])
    return x1, x2

def run(data, model):
    """

    Input:
        Data: the output of process_data function.
        Model: your model.
    Output:
        prediction: the predicted label(plate) of the data, an int value.

    """
    x1, x2 = data
    with torch.no_grad():
        outputs = model(x1, x2)
    _, predicted = torch.max(outputs.data, 1)
    sum = torch.sum(predicted).cpu().detach().numpy()
    return int(sum > len(x1)/2)


model = load_model()
s = 0
for d, l in zip(validate,label):
    data = process_data(d[0],d[1])
    prd = run(data,model)
    if l==prd:
        s+=1
print(s/len(validate))
