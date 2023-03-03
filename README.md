## 显卡性能基准测试

测试不同显卡在不同网络模型上的基准推理/训练速度（非正规）

ATTENTION!!! 为了测试尽量准确，请确保测试期间没有其他负载

测试FLOPs默认使用224*224*3图像作为输入
TIME EACH STEP是稳定之后的时间，此时间仅供参考，通常仅有前两位是准确的

以下为单卡测试结果
对于多卡并行，并行技术不同，效率会有非常大的区别

cmd：```python TrainingBenchmark.py {-m} {-b} {-d}```
* ```-m```: Model Name
* ```-b```: Batch Size
* ```-d```: Dataset Name

| MODEL       | DEVICE           | PARAMs/MB | DATASET | FLOPs/MB   | BATCH_SIZE | TIME EACH STEP / s | TIME EACH EPOCH / s |
|-------------|------------------|-----------|---------|------------|------------|--------------------|---------------------|
| AlexNet     | Nvidia Tesla P40 | -         | CIFAR10 | -          | -          | -                  | -                   |
| ResNet18    | Nvidia Tesla P40 | 10.6636   | CIFAR10 | 1734.3130  | 64         | 0.0191             | 24.8226             |
| ResNet34    | Nvidia Tesla P40 | 20.3035   | CIFAR10 | 3500.7056  | 64         | 0.0318             | 35.5037             |
| ResNet50    | Nvidia Tesla P40 | 22.4385   | CIFAR10 | 3919.1104  | 64         | 0.0447             | 45.8988             |
| ResNet101   | Nvidia Tesla P40 | 40.5509   | CIFAR10 | 7469.1221  | 64         | 0.0899             | 82.5451             |
| ResNet152   | Nvidia Tesla P40 | 55.4698   | CIFAR10 | 11021.4307 | 64         | 0.1298             | 115.3972            |
| DenseNet121 | Nvidia Tesla P40 | 6.6415    | CIFAR10 | 2731.8315  | 64         | 0.0975             | 88.6472             |
| DenseNet161 | Nvidia Tesla P40 | 25.2667   | CIFAR10 | 7424.0280  | 64         | 0.1343             | 119.2757            |
| DenseNet169 | Nvidia Tesla P40 | 11.9220   | CIFAR10 | 3238.9865  | 64         | 0.1402             | 123.6701            |
| DenseNet201 | Nvidia Tesla P40 | 17.2731   | CIFAR10 | 4137.9642  | 64         | 0.1718             | 149.2157            |
| VGG11       | Nvidia Tesla P40 | 122.8402  | CIFAR10 | 7252.7505  | 64         | 0.0441             | 45.1731             |
| VGG13       | Nvidia Tesla P40 | 123.0162  | CIFAR10 | 10780.7505 | 64         | 0.0495             | 49.6735             |
| VGG16       | Nvidia Tesla P40 | 128.0799  | CIFAR10 | 14749.7505 | 64         | 0.0549             | 54.0986             |
| VGG19       | Nvidia Tesla P40 | 133.1436  | CIFAR10 | 18718.7505 | 64         | 0.0600             | 58.5536             |
| VGG11BN     | Nvidia Tesla P40 | 122.8455  | CIFAR10 | 7266.9146  | 64         | 0.0454             | 45.8868             |
| VGG13BN     | Nvidia Tesla P40 | 123.0218  | CIFAR10 | 10804.1021 | 64         | 0.0518             | 51.6268             |
| VGG16BN     | Nvidia Tesla P40 | 128.0880  | CIFAR10 | 14775.5903 | 64         | 0.0570             | 55.9433             |
| VGG19BN     | Nvidia Tesla P40 | 133.1541  | CIFAR10 | 18747.0786 | 64         | 0.0630             | 60.8881             |
| Swin_s      | Nvidia Tesla P40 | 46.5822   | CIFAR10 | 8361.6372  | 64         | 0.1294             | 115.8975            |
| Swin_b      | Nvidia Tesla P40 | 82.7346   | CIFAR10 | 14750.3496 | 64         | 0.1830             | 159.9713            |
| Swin_t      | Nvidia Tesla P40 | 26.2518   | CIFAR10 | 4299.6138  | 64         | 0.0672             | 65.0438             |
| SwinV2_s    | Nvidia Tesla P40 | 46.7073   | CIFAR10 | 9270.5764  | 64         | 0.1933             | 168.2630            |
| SwinV2_b    | Nvidia Tesla P40 | 82.8897   | CIFAR10 | 16290.5107 | 64         | 0.2703             | 231.5499            |
| SwinV2_t    | Nvidia Tesla P40 | 26.3121   | CIFAR10 | 4725.5061  | 64         | 0.0973             | 89.5362             |
| VIT_b_32    | Nvidia Tesla P40 | -         | CIFAR10 | -          | -          | -                  | -                   |
| VIT_b_16    | Nvidia Tesla P40 | -         | CIFAR10 | -          | -          | -                  | -                   |
| VIT_h_14    | Nvidia Tesla P40 | -         | CIFAR10 | -          | -          | -                  | -                   |
| VIT_l_16    | Nvidia Tesla P40 | -         | CIFAR10 | -          | -          | -                  | -                   |
| VIT_l_32    | Nvidia Tesla P40 | -         | CIFAR10 | -          | -          | -                  | -                   |
| SwinV2_b    | Nvidia Tesla P40 | 82.8897   | CIFAR10 | 16290.5107 | 128        | 0.4749             | 204.8639            |
