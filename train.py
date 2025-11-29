import torch.optim as optim
import kornia.augmentation as K
from ueraser import UEraser, UEraser_jpeg
import os
import argparse
import torch.nn.functional as F
from madrys import MadrysLoss
from nets import *
from utils import *
import wandb
from tqdm import tqdm
from torchvision import models
from models.build_model import create_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "imagenetsubset", "webfacesubset"])
    parser.add_argument(
        "--data-path", type=str, default=None,
    )
    parser.add_argument(
        "--test-data-path", type=str, default=None,
    )
    parser.add_argument("--type", default=None)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")

    parser.add_argument("--clean", default=False, action="store_true")
    parser.add_argument("--cutout", default=False, action="store_true")
    parser.add_argument("--cutmix", default=False, action="store_true")
    parser.add_argument("--mixup", default=False, action="store_true")
    parser.add_argument("--rnoise", default=False, action="store_true")
    parser.add_argument("--pure", default=False, action="store_true")
    parser.add_argument("--jpeg", default=False, action="store_true")
    parser.add_argument("--jpeg-rate", default=10, type=int) 

    parser.add_argument("--bdr", default=False, action="store_true")
    parser.add_argument("--gray", default=False, action="store_true")
    parser.add_argument("--gaussian", default=False, action="store_true")
    parser.add_argument("--nodefense", default=False, action="store_true")

    parser.add_argument("--pretrained", default=False, action="store_true")

    parser.add_argument("--ueraser", default=False, action="store_true")
    parser.add_argument(
        "--repeat_epoch",
        default=300,
        type=int,
        help="0 for -lite / 50 for UEraser / 300 for -max",
    )

    parser.add_argument("--at", default=False, action="store_true")
    parser.add_argument("--at_eps", default=8 / 255,
                        type=float, help="noise budget")
    parser.add_argument(
        "--at_type", default="L_inf", type=str, help="noise type, [L_inf, L_2]"
    )

    parser.add_argument(
        "--arch", default="r18", type=str, help="r18, r50, se18, mv2, de121, vit, cait"
    )

    parser.add_argument("--gpu-id", "-g", default=0, type=int)

    parser.add_argument("--save", default=False, action="store_true")
    parser.add_argument("--save-path", )

    parser.add_argument("--dilute-path", default=None)

    args = parser.parse_args()
    if args.save and args.save_path is None:
        raise ValueError("Please specify save path")
    return args

def train(model, trainloader, optimizer, criterion, device, epoch, args):
    # model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    acc = 0
    if args.cutmix:
        cutmix = K.RandomCutMixV2(data_keys=["input", "class"])
    elif args.mixup:
        mixup = K.RandomMixUpV2(data_keys=["input", "class"])

    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        if args.cutmix or args.mixup:
            if args.cutmix:
                inputs, targets = cutmix(inputs, targets)
                targets = targets.squeeze(0)
            else:
                inputs, targets = mixup(inputs, targets)
            outputs = model(inputs)
            loss = loss_mix(targets, outputs)
            loss.backward()
            optimizer.step()
            total += targets.size(0)
            acc += torch.sum(acc_mix(targets, outputs))
            continue
        elif args.ueraser:
            if args.type == "tap" or args.type == "ar":
                U = UEraser_jpeg
            else:
                U = UEraser
            result_tensor = torch.empty((5, inputs.shape[0])).to(device)
            if epoch < args.repeat_epoch:
                for i in range(5):
                    images_tmp = U(inputs)
                    output_tmp = model(images_tmp)
                    loss_tmp = F.cross_entropy(
                        output_tmp, targets, reduction="none")
                    result_tensor[i] = loss_tmp
                outputs = output_tmp
                max_values, _ = torch.max(result_tensor, dim=0)
                loss = torch.mean(max_values)
            else:
                inputs = U(inputs)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            continue
        elif args.at:
            outputs, loss = MadrysLoss(epsilon=args.at_eps, distance=args.at_type)(
                model, inputs, targets, optimizer
            )
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        wandb.log({"train_loss": loss.item()})
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    if args.cutmix or args.mixup:
        avg_train_acc = acc * 100.0 / total
    else:
        avg_train_acc = correct * 100.0 / total
    return avg_train_acc


def test(model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    avg_test_acc = correct * 100.0 / total
    return avg_test_acc


def main():
    transform_train = aug_train(
        args.dataset, args.jpeg, args.gray, args.bdr, args.gaussian, args.cutout, args
    )
    train_set, test_set = get_dataset(args, transform_train, args.data_path, args.test_data_path)

    train_loader, test_loader = get_loader(args, train_set, test_set)

    if args.dataset == "cifar100" or args.dataset == "webfacesubset" or args.dataset == "imagenetsubset":
        num_classes = 100
    else:
        num_classes = 10

    if args.arch == "r18":
        if "cifar" in args.dataset:
            model = model = ResNet18(num_classes)
        else:            
            if args.pretrained:
                model = models.resnet18(weights='DEFAULT')
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            else:
                model = models.resnet18(weights=None, num_classes=num_classes)

    elif args.arch == "r50":
        model = ResNet50(num_classes)
    elif args.arch == "se18":
        model = SENet18(num_classes)
    elif args.arch == "mv2":
        model = MobileNetV2(num_classes)
    elif args.arch == "de121":
        model = DenseNet121(num_classes)
    elif args.arch == "vit":
        model = Vit_cifar(num_classes)
    elif args.arch == "cait":
        model = create_model(32, num_classes, args)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    if args.arch == "vit" or args.arch == "cait":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
        )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[80, 100], gamma=0.1, last_epoch=-1, verbose=False
    )

    for epoch in range(args.epochs):
        train_acc = train(
            model, train_loader, optimizer, criterion, device, epoch, args
        )
        test_acc = test(model, test_loader, criterion, device)
        wandb.log({"train_acc": train_acc, "test_acc": test_acc, "epoch": epoch})
        scheduler.step()
    if args.save:
        save_dir = os.path.dirname(args.save_path)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), args.save_path)

if __name__ == "__main__":
    args = get_args()
    log_name = args.data_path if args.data_path is not None else args.dataset
    wandb.init(project="BridgePure-eval", name=log_name, config=vars(args))
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    main()