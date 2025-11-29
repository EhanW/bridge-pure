import torch
from PIL import Image
import os
import sys
import time
import random
from io import BytesIO
import PIL
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

class NPZDataset(Dataset):
    def __init__(self, root, train, transform=None, data_path=None, indexed=False):
        """
        Args:
            npz_file (string): Path to the npz file.
            transform (callable, optional): Optional transform to be applied
                on a sample (such as image normalization, augmentation, etc.).
        """
        if data_path is not None:
            self.data = np.load(data_path)['data']
            self.targets = np.load(data_path)['targets']
        else:
            if train:
                npz_file = os.path.join(root, 'train_dataset.npz')
            else:
                npz_file = os.path.join(root, 'test_dataset.npz')
            self.dataset = np.load(npz_file)
            self.data = self.dataset['data']  # Assumes the npz file has 'data' key for images
            self.targets = self.dataset['targets']  # Assumes the npz file has 'targets' key for labels
        self.transform = transform

        self.indexed = indexed
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        if self.indexed:
            return image, label, idx
        return image, label
    
def get_data_dir(dataset):
    if os.path.exists('./data'):
        root_dir='./data'
    elif os.path.exists('/datasets/'):
        root_dir='/datasets'
    elif os.path.exists('../../data'):
        root_dir='../../data'
    else:
        raise ValueError('No such directory')
    if 'cifar' in dataset:
        return root_dir
    
    elif 'imagenet' in dataset:
        data_dir = os.path.join(root_dir, dataset)
        return data_dir
    
    elif 'webface' in dataset:
        data_dir = os.path.join(root_dir, dataset)
        return data_dir
    


class CIFAR10Index(datasets.CIFAR10):
    def __init__(self, indexed=False, data_path=None, **kwargs):
        super().__init__(**kwargs)
        if data_path is not None:
            self.data = np.load(data_path)['data']
            self.targets = np.load(data_path)['targets']
        self.indexed = indexed

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.indexed:
            return img, target, idx
        return img, target

class CIFAR100Index(datasets.CIFAR100):
    def __init__(self, indexed=False, data_path=None, **kwargs):
        super().__init__(**kwargs)
        if data_path is not None:
            self.data = np.load(data_path)['data']
            self.targets = np.load(data_path)['targets']
        self.indexed = indexed
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.indexed:
            return img, target, idx
        return img, target


TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time

term_width = 80


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append("  Step: %s" % format_time(step_time))
    L.append(" | Tot: %s" % format_time(tot_time))
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f


# def JPEGcompression(image, rate=10):
#     outputIoStream = BytesIO()
#     image.save(outputIoStream, "JPEG", quality=rate, optimice=True)
#     outputIoStream.seek(0)
#     return Image.open(outputIoStream)

class JPEGcompression:
    def __init__(self, rate=10):
        self.rate = rate
    
    def __call__(self, img):
        outputIoStream = BytesIO()
        img.save(outputIoStream, "JPEG", quality=self.rate, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)



def aug_train(dataset, jpeg, grayscale, bdr, gaussian, cutout, args):
    transform_train = transforms.Compose([])

    if bdr:
        transform_train.transforms.append(
            transforms.RandomPosterize(bits=2, p=1))
    if grayscale:
        transform_train.transforms.append(transforms.Grayscale(3))
    if jpeg:
        transform_train.transforms.append(transforms.Lambda(JPEGcompression(rate=args.jpeg_rate)))
    if gaussian:
        transform_train.transforms.append(
            transforms.GaussianBlur(3, sigma=0.1))

    if dataset == "imagenetsubset":
        transform_train.transforms.append(transforms.RandomResizedCrop(224))
        transform_train.transforms.append(transforms.RandomHorizontalFlip())
        transform_train.transforms.append(transforms.ToTensor())
    elif dataset == "cifar10":
        transform_train.transforms.append(transforms.RandomCrop(32, padding=4))
        transform_train.transforms.append(transforms.RandomHorizontalFlip())
        transform_train.transforms.append(transforms.ToTensor())
    elif dataset == "cifar100":
        transform_train.transforms.append(transforms.RandomCrop(32, padding=4))
        transform_train.transforms.append(transforms.RandomHorizontalFlip())
        transform_train.transforms.append(transforms.ToTensor())
    elif dataset == "svhn":
        transform_train.transforms.append(transforms.ToTensor())
    elif dataset == "webfacesubset":
        transform_train.transforms.append(transforms.RandomHorizontalFlip())
        transform_train.transforms.append(transforms.ToTensor())
    if cutout:
        transform_train.transforms.append(Cutout(16))
    return transform_train


def get_dataset(args, transform_train, data_path, test_data_path=None):
    transform_test = transforms.Compose([])
    transform_test.transforms.append(transforms.ToTensor())

    root_dir = get_data_dir(args.dataset)
    if args.dataset == "cifar10":
        train_dataset = CIFAR10Index(root=root_dir, train=True, download=True, transform=transform_train, data_path=data_path)
        test_dataset = datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform_test)
    elif args.dataset == "cifar100":
        train_dataset = CIFAR100Index(root=root_dir, train=True, download=True, transform=transform_train, data_path=data_path)
        test_dataset = datasets.CIFAR100(root=root_dir, train=False, download=True, transform=transform_test)
    elif args.dataset == "webfacesubset" or args.dataset == "imagenetsubset":
        train_dataset = NPZDataset(root=root_dir, train=True, transform=transform_train, data_path=data_path)
        test_dataset = NPZDataset(root=root_dir, train=False, transform=transform_test)
    else:
        raise ValueError("Valid type and dataset.")
    return train_dataset, test_dataset


def get_loader(args, train_dataset, test_dataset):
    if 'cifar' in args.dataset:
        num_workers = 4
    else:
        num_workers = 8

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.bs, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader

def loss_mix(y, logits):
    criterion = F.cross_entropy
    loss_a = criterion(logits, y[:, 0].long(), reduction="none")
    loss_b = criterion(logits, y[:, 1].long(), reduction="none")
    return ((1 - y[:, 2]) * loss_a + y[:, 2] * loss_b).mean()


def acc_mix(y, logits):
    pred = torch.argmax(logits, dim=1).to(y.device)
    return (1 - y[:, 2]) * pred.eq(y[:, 0]).float() + y[:, 2] * pred.eq(y[:, 1]).float()