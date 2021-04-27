import pickle
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import datetime
from tqdm import tqdm

long_mean, long_std, lat_mean, lat_std = (114.0471251204773, 0.09265583326259941, 22.555874215757328, 0.04737859070837168)

def load_data(traj_len, num_drivers, device):
    X = {}
    data = pickle.load(open('project_4_train.pkl','rb'))
    for driver in tqdm(range(num_drivers)):
        X[driver] = torch.cat([torch.stack([featurize(subtraj, device) for subtraj in split_traj(traj, traj_len)]) for traj in data[driver]])
    return X

def featurize(traj_raw, device):
    return torch.tensor(list(map(featurize_row, traj_raw)), dtype = torch.float, device = device)

def featurize_row(row):
    if len(row) == 6:
        row = row[1:]
    norm_long = (row[0] - long_mean) / long_std
    norm_lat = (row[1] - lat_mean) / lat_std
    return np.concatenate(([norm_long, norm_lat], sin_cos(row[2] / 86400), [row[3]], convert_time(row[4]))).astype(float)

def sin_cos(n):
    theta = 2 * np.pi * n
    return (np.sin(theta), np.cos(theta))

def convert_time(time):
    d = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return list(sum(map(lambda t: sin_cos(t), [(d.month - 1)/ 12, (d.day - 1) / months[d.month - 1], d.weekday() / 7, d.hour / 24, d.minute / 60, d.second / 60]), ()))

def split_traj(traj, max_len):
    result = []
    subtraj = [traj[0]]
    for t in traj:
        if len(subtraj) < max_len:
            subtraj.append(t)
        else:
            result.append(subtraj)
            subtraj = [t]
    return result

def train_test_split(dataset, test_split, batch_size):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_size = int(test_split * dataset_size)
    test_idx = np.random.choice(indices, size=test_size, replace=False)
    train_idx = list(set(indices) - set(test_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = DataLoader(dataset, batch_size = batch_size, sampler=train_sampler)
    testloader = DataLoader(dataset, batch_size = batch_size, sampler=test_sampler)

    return trainloader, testloader

if __name__ == '__main__':
    print("Utils test:")
    X = load_data(64, 5, torch.device("cuda"))
    print(X[0][0][0][0], X[0][0][0].shape)
