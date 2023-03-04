import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

parser.add_argument("-m", "--model_type", type=str, default="AlexNet", choices=model_list.keys())
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-s", "--input_size", type=int, default=32)
parser.add_argument("--max_step", type=int, default=800)


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
    log_file_name = "./log/{}_{}{}_{}_{}.txt".format(args.model_type, args.input_size, device_name_str, args.batch_size, random_str)
    with open(log_file_name, "a") as f: f.write(str(time.time()) + "\n")

    # optimizing
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    begin_time = time.time()
    time_step_list = list()
    images = torch.rand([args.batch_size, 3, args.input_size, args.input_size]).to(device)
    labels = torch.randint(0, 10, [args.batch_size, ]).to(device)

    net.train()
    for i in range(args.max_step):
        step_begin_time = time.time()

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        step_end_time = time.time()
        with open(log_file_name, "a") as f:
            time_step_list.append(step_end_time - step_begin_time)
            log_str = "Step: [{}/{}], Avg time for each step {:.4f}s" \
                .format(i + 1, args.max_step, time_step_list[-1])
            print(log_str)
            f.write(log_str + "\n")

    print("-------------------------\n--- PARAMs & FLOPs --- \n-------------------------")
    test_tensor = torch.rand([1, 3, 224, 224]).to(device)
    flops = FlopCountAnalysis(net, test_tensor).total()
    params = parameter_count(net)[""]
    with open(log_file_name, "a") as f:
        log_str = "PARAMS: {:.4f}MB, FLOPS: {:.4f}MB\n".format(params / (1024 ** 2), flops / (1024 ** 2))
        print(log_str)
        f.write(log_str)

    print("-------------------------\n----- DEVICE ----- \n-------------------------")
    device_name = None
    for _id in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(_id)
        info = f"CUDA:{_id} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"
        if _id == 0:
            device_name = p.name
        print(info)
        with open(log_file_name, "a") as f:
            f.write(info)

    print("-------------------------\n----- EACH STEP ----- \n-------------------------")
    with open(log_file_name, "a") as f:
        avg_step_str = "Avg time for each step {:.4f}s".format(np.mean(time_step_list))
        print(avg_step_str)
        f.write(avg_step_str + "\n")

    with open("./README.md", "a") as f:
        str_1 = f"\n|{args.model_type}|{device_name}|{params / (1024 ** 2):.4f}|{args.input_size}"
        str_2 = f"|{flops / (1024 ** 2):.4f}|{args.batch_size}|{np.mean(time_step_list):.4f}|"
        f.write(str_1 + str_2)
