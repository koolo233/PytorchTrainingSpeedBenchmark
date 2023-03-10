import os
import sys
import time
import random
import warnings
import argparse
import numpy as np
from fvcore.nn import FlopCountAnalysis, parameter_count

from utils.models import torchvision_model_dict
from utils.log import generate_random_str

import torch
from torch import nn
import torch.optim as optim
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.utilities.warnings import PossibleUserWarning

warnings.filterwarnings("ignore", category=PossibleUserWarning)


parser = argparse.ArgumentParser(description='Params of Benchmark')

parser.add_argument("-m", "--model_type", type=str, default="AlexNet")
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-s", "--input_size", type=int, default=32)
parser.add_argument("--gpu_ids", type=str, default="0")
parser.add_argument("--max_step", type=int, default=800)


class MyDataSet(Dataset):

    def __init__(self, batch_size, input_size, max_step):
        super().__init__()
        self.images = torch.rand([batch_size, 3, input_size, input_size])
        self.labels = torch.randint(0, 10, [batch_size, ])
        self.max_step = max_step

    def __len__(self):
        return self.max_step

    def __getitem__(self, item):
        return self.images, self.labels


class BenchmarkSystem(LightningModule):

    def __init__(self, args):
        super().__init__()

        # model
        self.model_type = args.model_type
        model_dict, self.version_str_list = torchvision_model_dict()

        if self.model_type not in list(model_dict.keys()):
            raise ValueError(f"{self.model_type} is not supported.")

        self.net = model_dict[self.model_type](num_classes=10)

        # ------------
        # --- data ---
        # ------------
        self.dataset = MyDataSet(args.batch_size, args.input_size, args.max_step)

        # ----------------------
        # --- loss functions ---
        # ----------------------
        self.criterion = nn.CrossEntropyLoss()

        self.fit_begin_time = None
        self.fit_end_time = None
        self.train_begin_time = None
        self.train_end_time = None

        # --------------
        # --- others ---
        # --------------
        self.max_step = args.max_step
        self.gpus = args.gpus
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.log_file_name = args.log_file_name
        self.model_type = args.model_type

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            shuffle=False,
            num_workers=4,
            batch_size=1,
            pin_memory=True
        )

    def forward(self, batch_imgs):
        res = self.net(batch_imgs)
        return res

    def configure_optimizers(self):
        optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_nb):
        images, labels = batch
        images = torch.squeeze(images, dim=0)
        labels = torch.squeeze(labels, dim=0)
        if self.model_type == "InceptionV3":
            outputs, auxs = self(images)
        elif self.model_type == "GoogleNet":
            outputs, auxs1, auxs2 = self(labels)
        else:
            outputs = self(images)
        loss = self.criterion(outputs, labels)
        return loss

    def on_fit_start(self):
        if self.global_rank == 0:
            self.fit_begin_time = time.time()

    def on_fit_end(self):
        if self.global_rank == 0:
            self.fit_end_time = time.time()

            version_str = ".".join(self.version_str_list)
            avg_step_time = (self.fit_end_time - self.fit_begin_time) / self.max_step

            print("-------------------------\n--- PARAMs & FLOPs --- \n-------------------------")
            test_tensor = torch.rand([2, 3, self.input_size, self.input_size]).to(self.device)
            flops = FlopCountAnalysis(self.net, test_tensor).total() / 2
            params = parameter_count(self.net)[""]
            with open(self.log_file_name, "a") as f:
                log_str = "PARAMS: {:.4f}MB, FLOPS: {:.4f}\n".format(params / (1024 ** 2), flops / (1024 ** 2))
                print(log_str)
                f.write(log_str)

            print("-------------------------\n----- DEVICE ----- \n-------------------------")
            device_name_dict = dict()
            for _id in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(_id)
                info = f"CUDA:{_id} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"
                if p.name not in device_name_dict:
                    device_name_dict[p.name] = 1
                else:
                    device_name_dict[p.name] += 1
                print(info)
                with open(self.log_file_name, "a") as f:
                    f.write(info)

            device_name = ""
            for key, item in device_name_dict.items():
                device_name += f"{item} X {key} "

            print("-------------------------\n----- EACH STEP ----- \n-------------------------")
            with open(self.log_file_name, "a") as f:
                avg_step_str = "Avg time for each step {:.4f}s".format(avg_step_time)
                print(avg_step_str)
                f.write(avg_step_str + "\n")

            with open("./README.md", "a") as f:
                str_1 = f"\n|{self.model_type}|{device_name}|{params / (1024 ** 2):.4f}|{args.input_size}"
                str_2 = f"|{flops / (1024 ** 2):.4f}|{self.batch_size}|{avg_step_time:.4f}|{version_str}|"
                f.write(str_1 + str_2)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):

    setup_seed(1024)

    pbar = TQDMProgressBar(refresh_rate=1)

    device_name_str = ""
    for _id in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(_id)
        device_name_str += f"_{p.name}"
    random_str = generate_random_str()
    log_file_name = "./log/{}_{}{}_{}_{}.txt".format(args.model_type, args.input_size, device_name_str, args.batch_size,
                                                     random_str)

    num_gpus = len(args.gpu_ids.split(","))

    if sys.platform == "win32":
        ddp_backend = "gloo"
    else:
        ddp_backend = None

    trainer = Trainer(
        max_steps=args.max_step,
        max_epochs=-1,
        callbacks=[pbar],
        limit_val_batches=0.0,
        enable_model_summary=False,
        accelerator="auto",
        devices=num_gpus,
        benchmark=True,
        profiler="simple" if num_gpus == 1 else None,
        strategy=DDPStrategy(find_unused_parameters=False, process_group_backend=ddp_backend) if num_gpus > 1 else None,
    )

    args.max_step = args.max_step * num_gpus
    args.gpus = args.gpu_ids.split(",")
    args.log_file_name = log_file_name

    system = BenchmarkSystem(args)

    trainer.fit(system)


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    main(args=args)
