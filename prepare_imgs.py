import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from torchvision.datasets import CIFAR10, CIFAR100
from utils import get_data_dir

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--attack', type=str, default='lsp',)
    parser.add_argument('--num-training-data', type=int, default=1000)
    parser.add_argument('--num-purifying-data', type=int, default=40000)
    parser.add_argument('--jpeg-quality', type=int, default=10)

    parser.add_argument('--beta-max', type=float, default=0)
    parser.add_argument('--get-baseline', action='store_true', default=False)
    return parser.parse_args()


def generate_gaussian_noise(x, b):
    if b == 0:
        return x
    b = torch.tensor(b)
    noise = torch.randn_like(x)
    noisy_x = (1 - b).sqrt() * x + b.sqrt() * noise
    return noisy_x

def gen_aligned_images_with_guassian(dataset_name, attack, num_training_data, num_purifying_data=40000, beta_max=0.02):
    root_dir = get_data_dir(dataset_name)
    if dataset_name == 'cifar10':
        trainset = CIFAR10(root=root_dir, train=True, download=True, transform=None)
    elif dataset_name == 'cifar100':
        trainset = CIFAR100(root=root_dir, train=True, download=True, transform=None)
    else:
        raise ValueError('No such dataset')

    if attack != 'clean':
        poison = torch.load(f'./data/{attack}-{dataset_name}.pt', map_location='cpu').to(torch.float32)
    else:
        poison = None

    num_train = num_training_data
    num_val = num_purifying_data
    num_test = len(trainset) - num_train - num_val

    save_dir = f'./images/{dataset_name}/{attack}_gaussian_{beta_max}/{num_train}_{num_val}'
    for n in ['train', 'val', 'test']:
        os.makedirs(os.path.join(save_dir, n), exist_ok=True)
    print(save_dir)
    data_40000 = []
    targets_40000 = []
    for i, (image, target) in tqdm(enumerate(trainset), total=num_train+num_val+num_test):
        clean_image = np.asarray(image).copy()
        clean_data = torch.from_numpy(clean_image).permute(2, 0, 1).to(torch.float32)/255
        if poison is not None:
            clean_data = clean_data + poison[i]
            clean_data = torch.clamp(clean_data, 0, 1)
        clean_data_normalized = 2*clean_data-1

        if i < num_val:
            noisy_data_normalized = generate_gaussian_noise(clean_data_normalized, beta_max)
            noisy_data = (noisy_data_normalized+1)/2
            noisy_data = noisy_data.permute(1, 2, 0).numpy()
            noisy_data = np.clip(noisy_data, 0, 1)
            noisy_image = (noisy_data*255).astype(np.uint8)
            p2c_image = np.concatenate([noisy_image, clean_image], axis=1)
            p2c_image = Image.fromarray(p2c_image)
            data_40000.append(noisy_image)
            targets_40000.append(target)
            p2c_image.save(os.path.join(save_dir, f'val/{i}.png'))

        elif i < num_train + num_val:
            noisy_data_normalized = generate_gaussian_noise(clean_data_normalized, beta_max)
            noisy_data = (noisy_data_normalized+1)/2
            noisy_data = noisy_data.permute(1, 2, 0).numpy()
            noisy_data = np.clip(noisy_data, 0, 1)
            noisy_image = (noisy_data*255).astype(np.uint8)
            p2c_image = np.concatenate([noisy_image, clean_image], axis=1)
            p2c_image = Image.fromarray(p2c_image)
            p2c_image.save(os.path.join(save_dir, f'train/{i}.png'))


def get_clean_data_targets(dataset_name, num_purifying_data):
    num_purifying_data=num_purifying_data
    root_dir = get_data_dir(dataset_name)
    if dataset_name == 'cifar10':
        dataset = CIFAR10(root=root_dir, train=True, download=True, transform=None)
    elif dataset_name == 'cifar100':
        dataset = CIFAR100(root=root_dir, train=True, download=True, transform=None)
    else:
        raise ValueError('No such dataset')
    data = dataset.data[:num_purifying_data]
    targets = np.array(dataset.targets)[:num_purifying_data]
    save_dir = f'images/{dataset_name}/clean/{num_purifying_data}_val_data_targets'
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, 'data_targets.npz'), data=data, targets=targets)

def get_poisoned_data_targets(dataset_name, attack, num_training_data, num_purifying_data):
    root_dir = get_data_dir(dataset_name)
    if dataset_name == 'cifar10':
        dataset = CIFAR10(root=root_dir, train=True, download=True, transform=None)
    elif dataset_name == 'cifar100':
        dataset = CIFAR100(root=root_dir, train=True, download=True, transform=None)
    else:
        raise ValueError('No such dataset')
    
    num_train = num_training_data
    num_val = num_purifying_data
    num_test = len(dataset) - num_train - num_val
    dataset.data = dataset.data[:num_val]
    dataset.targets = dataset.targets[:num_val]

    poison = torch.load(f'./data/{attack}-{dataset_name}.pt', map_location='cpu')
    save_dir = f'images/{dataset_name}/{attack}/{num_val}_val_data_targets'
    os.makedirs(save_dir, exist_ok=True)

    data = []
    targets = []
    for i, (image, label) in tqdm(enumerate(dataset), total=num_val):
        clean_image = np.asarray(image)
        delta  = poison[i].squeeze().mul(255).numpy().transpose(1,2,0)
        poisoned_image = (delta+clean_image).clip(0,255).astype(np.uint8)
        data.append(poisoned_image)
        targets.append(label)
    data = np.stack(data)
    targets = np.array(targets)
    np.savez(os.path.join(save_dir, 'data_targets.npz'), data=data, targets=targets)


if __name__ == '__main__':
    args = get_args()
    if args.get_baseline:
        get_clean_data_targets(dataset_name=args.dataset, num_purifying_data=args.num_purifying_data)
        get_poisoned_data_targets(dataset_name=args.dataset, attack=args.attack, num_training_data=args.num_training_data, num_purifying_data=args.num_purifying_data)
    else:
        gen_aligned_images_with_guassian(dataset_name=args.dataset, attack=args.attack, num_training_data=args.num_training_data, num_purifying_data=args.num_purifying_data, beta_max=args.beta_max)
