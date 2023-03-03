## 显卡性能基准测试

### TODO
- [x] 基准测试结果自动填写
- [ ] log批量处理
- [ ] 对比结果绘制
- [ ] 同设备多log处理

测试不同显卡在不同网络模型上的基准推理/训练速度（非正规）

ATTENTION!!! 为了测试尽量准确，请确保测试期间没有其他负载

测试FLOPs默认使用32 x 32 x 3图像作为输入

TIME EACH STEP是稳定之后的时间，此时间仅供参考，通常仅有前两位是准确的

以下为单卡测试结果

对于多卡并行，并行技术不同，效率会有非常大的区别

cmd：```python TrainingBenchmark.py {-m} {-b} {-s}```
* ```-m```: Model Name
* ```-b```: Batch Size
* ```-s```: Input Size

| MODEL       | DEVICE           | PARAMs/MB | INPUT SIZE | FLOPs/MB | BATCH_SIZE | TIME EACH STEP / s |
|-------------|------------------|-----------|------------|----------|------------|--------------------|
|ResNet18|Tesla P40|10.6636|32|1734.3130|64|0.0221|
|ResNet50|Tesla P40|22.4385|32|3919.1104|64|0.0549|