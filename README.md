## 显卡性能基准测试

### TODO
- [x] 基准测试结果自动填写
- [x] 采用全虚拟数据，尽量排除CPU干扰
- [ ] log批量处理
- [ ] 对比结果绘制
- [ ] 同设备多log处理

测试不同显卡在不同网络模型上的基准推理/训练速度（非正规）

ATTENTION!!! 为了测试尽量准确，请确保测试期间没有其他负载

测试FLOPs默认使用224 x 224 x 3图像作为输入

TIME EACH STEP是稳定之后的时间，此时间仅供参考，通常仅有前两位是准确的

以下为单卡测试结果

对于多卡并行，并行技术不同，效率会有非常大的区别

cmd：```python TrainingBenchmark.py {-m} {-b} {-s}```
* ```-m```: Model Name
* ```-b```: Batch Size
* ```-s```: Input Size

若需要指定显卡，请使用```CUDA_VISIBLE_DEVICES={id}```

例如，若想指定1卡用于训练：```CUDA_VISIBLE_DEVICES=1 TrainingBenchmark.py {-m} {-b} {-s}```

| MODEL       | DEVICE           | PARAMs/MB | INPUT SIZE | FLOPs/MB | BATCH_SIZE | TIME EACH STEP / s |
|-------------|------------------|-----------|------------|----------|------------|--------------------|
|VIT_b_32|Tesla P40|83.4111|224|4208.7803|64|0.2830|
|ResNet50|Tesla P40|22.4385|32|3950.9077|64|0.0490|
|ResNet101|Tesla P40|40.5509|32|7515.5620|64|0.0879|
|ResNet152|Tesla P40|55.4698|32|11085.9585|64|0.1235|
|ResNet18|Tesla P40|10.6636|32|1741.4189|64|0.0191|
|ResNet34|Tesla P40|20.3035|32|3511.4004|64|0.0308|