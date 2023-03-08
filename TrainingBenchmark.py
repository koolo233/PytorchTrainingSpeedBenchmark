import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from fvcore.nn import FlopCountAnalysis, parameter_count
from utils.models import torchvision_model_dict
from utils.log import generate_random_str


parser = argparse.ArgumentParser(description='Params of Benchmark')

parser.add_argument("-m", "--model_type", type=str, default="AlexNet")
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-s", "--input_size", type=int, default=32)
parser.add_argument("--gpu_ids", type=str, default="0")
parser.add_argument("--max_step", type=int, default=800)


if __name__ == "__main__":

    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    model_dict, version_str_list = torchvision_model_dict()
    version_str = ".".join(version_str_list)

    if args.model_type not in list(model_dict.keys()):
        raise ValueError(f"{args.model_type} is not supported.")

    net = model_dict[args.model_type](num_classes=10)
    print(args.model_type)

    if not os.path.exists("./log"):
        os.mkdir("./log")

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
    with open(log_file_name, "w") as f:
        f.write(str(time.time()) + "\n")
        f.write(version_str + "\n")

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
        if args.model_type == "InceptionV3":
            outputs, auxs = net(images)
        elif args.model_type == "GoogleNet":
            outputs, auxs1, auxs2 = net(images)
        else:
            outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        step_end_time = time.time()
        with open(log_file_name, "a") as f:
            time_step_list.append(step_end_time - step_begin_time)
            log_str = "Step: [{}/{}], Avg time for each step {:.4f}s" \
                .format(i + 1, args.max_step, np.mean(time_step_list))
            print(log_str)
            f.write(log_str + "\n")

    print("-------------------------\n--- PARAMs & FLOPs --- \n-------------------------")
    test_tensor = torch.rand([2, 3, args.input_size, args.input_size]).to(device)
    print(torch.typename(test_tensor))
    flops = FlopCountAnalysis(net, test_tensor).total() / 2
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
        str_2 = f"|{flops / (1024 ** 2):.4f}|{args.batch_size}|{np.mean(time_step_list):.4f}|{version_str}|"
        f.write(str_1 + str_2)
