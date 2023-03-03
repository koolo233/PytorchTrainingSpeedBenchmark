import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
import time
import string
import random
from fvcore.nn import FlopCountAnalysis, parameter_count

model_list = {
    "AlexNet": models.alexnet,
    "ResNet18": models.resnet18,
    "ResNet34": models.resnet34,
    "ResNet50": models.resnet50,
    "ResNet101": models.resnet101,
    "ResNet152": models.resnet152,
    "DenseNet121": models.densenet121,
    "DenseNet161": models.densenet161,
    "DenseNet169": models.densenet169,
    "DenseNet201": models.densenet201,
    "VGG11": models.vgg11,
    "VGG13": models.vgg13,
    "VGG16": models.vgg16,
    "VGG19": models.vgg19,
    "VGG11BN": models.vgg11_bn,
    "VGG13BN": models.vgg13_bn,
    "VGG16BN": models.vgg16_bn,
    "VGG19BN": models.vgg19_bn,
    "Swin_s": models.swin_s,
    "Swin_b": models.swin_b,
    "Swin_t": models.swin_t,
    "SwinV2_s": models.swin_v2_s,
    "SwinV2_b": models.swin_v2_b,
    "SwinV2_t": models.swin_v2_t,
    "VIT_b_32": models.vit_b_32,
    "VIT_b_16": models.vit_b_16,
    "VIT_h_14": models.vit_h_14,
    "VIT_l_16": models.vit_l_16,
    "VIT_l_32": models.vit_l_32
}

parser = argparse.ArgumentParser(description='Params of Benchmark')

parser.add_argument("-m", "--model_type", type=str, default="ResNet50", choices=model_list.keys())
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-d", "--dataset_name", type=str, default="CIFAR10", choices=["CIFAR10"])


def generate_random_str(length=30):
    # string.ascii_letters 大小写字母， string.digits 为数字
    characters_long = list(string.ascii_letters + string.digits)

    # 打乱字符串序列
    random.shuffle(characters_long)

    # picking random characters from the list
    password = []
    # 生成密码个数
    for b in range(length):
        password.append(random.choice(characters_long))

        # 打乱密码顺序
        random.shuffle(password)

    # 将列表转换为字符串并打印
    return "".join(password)


if __name__ == "__main__":

    args = parser.parse_args()

    # image_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize(input_size)])
    image_transforms = transforms.ToTensor()

    # load CIFA-10 data
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data/',
        train=True,
        transform=image_transforms,
        download=True)

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data/',
        train=False,
        transform=image_transforms,
        download=True)

    print('train_dataset = ', len(train_dataset))
    print('test_dataset = ', len(test_dataset))

    # set data loadser
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0)

    net = model_list[args.model_type](num_classes=10)
    print(args.model_type)

    # select device
    num_classes = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)

    device_name_str = ""
    for _id in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(_id)
        device_name_str += f"_{p.name}"
    random_str = generate_random_str()
    log_file_name = "./log/{}_{}{}_{}_{}.txt".format(args.model_type, args.dataset_name, device_name_str, args.batch_size, random_str)
    with open(log_file_name, "a") as f: f.write(str(time.time()) + "\n")

    # optimizing
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # training
    num_epochs = 2
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []

    begin_time = time.time()
    time_step_list = list()
    # training
    for epoch in range(num_epochs):
        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0

        # ====== train_mode ======
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            step_begin_time = time.time()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            train_acc += (outputs.max(1)[1] == labels).sum().item()
            loss.backward()
            optimizer.step()

            step_end_time = time.time()
            with open(log_file_name, "a") as f:
                time_step_list.append(step_end_time - step_begin_time)
                log_str = "Epoch [{}/{}], Step: [{}/{}], Avg time for each step {:.4f}s"\
                    .format(epoch + 1, num_epochs, i + 1, len(train_loader), time_step_list[-1])
                print(log_str)
                f.write(log_str + "\n")

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_acc / len(train_loader.dataset)

        # ====== val_mode ======
        net.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_val_loss = val_loss / len(test_loader.dataset)
        avg_val_acc = val_acc / len(test_loader.dataset)

        end_time = time.time()

        log_str = "Epoch [{}/{}], Time: [{:.4f}s], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"\
            .format(epoch + 1, num_epochs, end_time-begin_time, i + 1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc)

        print(log_str)
        with open(log_file_name, "a") as f:
            f.write(log_str + "\n")

    end_time = time.time()

    print("-------------------------\n--- PARAMs & FLOPs --- \n-------------------------")
    test_tensor = torch.rand([1, 3, 224, 224]).to(device)
    flops = FlopCountAnalysis(net, test_tensor).total()
    params = parameter_count(net)[""]
    with open(log_file_name, "a") as f:
        log_str = "PARAMS: {:.4f}MB, FLOPS: {:.4f}MB\n".format(params / (1024 ** 2), flops / (1024 ** 2))
        print(log_str)
        f.write(log_str)

    print("-------------------------\n----- DEVICE ----- \n-------------------------")
    for _id in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(_id)
        info = f"CUDA:{_id} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"
        print(info)
        with open(log_file_name, "a") as f:
            f.write(info)

    with open(log_file_name, "a") as f:
        avg_epoch_str = "Avg time for each epoch {:.4f}s".format((end_time - begin_time) / num_epochs)
        avg_step_str = "Avg time for each step {:.4f}s".format(np.mean(time_step_list))
        print(avg_epoch_str)
        print(avg_step_str)
        f.write(avg_epoch_str + "\n")
        f.write(avg_step_str + "\n")
