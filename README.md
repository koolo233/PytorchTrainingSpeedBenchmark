## 显卡性能基准测试

### TODO
- [x] 基准测试结果自动填写
- [x] 采用全虚拟数据，尽量排除CPU干扰
- [ ] log批量处理
- [ ] 对比结果绘制
- [ ] 同设备多log处理

测试不同显卡在不同网络模型上的基准推理/训练速度（非正规）

ATTENTION!!! 为了测试尽量准确，请确保测试期间没有其他负载

测试FLOPs默认使用单张训练图像作为输入，例如设定训练输入为224，则测试FLOPs时也使用224的尺寸

TIME EACH STEP是稳定之后的时间，此时间仅供参考

以下为单卡测试结果

对于多卡并行，并行技术不同，效率会有非常大的区别

cmd：```python TrainingBenchmark.py {-m} {-b} {-s}```
* ```-m```: Model Name
* ```-b```: Batch Size
* ```-s```: Input Size

若需要指定显卡，请使用```CUDA_VISIBLE_DEVICES={id}```

例如，若想指定1卡用于训练：```CUDA_VISIBLE_DEVICES=1 TrainingBenchmark.py {-m} {-b} {-s}```

| MODEL       | DEVICE           | PARAMs/MB | INPUT SIZE | FLOPs/MB | BATCH_SIZE | TIME EACH STEP / s | VERSION  |
|-------------|------------------|-----------|------------|----------|------------|--------------------|----------|
|VIT_b_32|Tesla P40|83.4111|224|4208.7803|64|0.2830|0.14|
|ResNet50|Tesla P40|22.4385|32|80.6499|64|0.0490|0.14|
|ResNet101|Tesla P40|40.5509|32|153.3979|64|0.0879|0.14|
|ResNet152|Tesla P40|55.4698|32|226.2632|64|0.1235|0.14|
|ResNet18|Tesla P40|10.6636|32|35.5439|64|0.0191|0.14|
|ResNet34|Tesla P40|20.3035|32|71.6660|64|0.0308|0.14|
|VIT_b_16|Tesla P40|81.8313|224|16767.4827|64|1.0535|0.14|
|VIT_l_32|Tesla P40|291.3672|224|14676.1602|64|0.8848|0.14|
|VIT_l_16|Tesla P40|289.2608|224|58749.3154|32|1.7269|0.14|
|SwinV2_t|Tesla P40|26.3121|224|4725.5061|64|0.7424|0.14|
|MaxVIT_t|Tesla P40|29.0039|224|5388.6119|64|1.0890|0.14|
|VIT_h_14|Tesla P40|601.5564|224|159645.7996|16|2.3230|0.14|
|AlexNet|Tesla P40|54.4022|224|677.2448|64|0.0427|0.14|
|VGG11|Tesla P40|122.8402|224|7252.7505|64|0.3236|0.14|
|VGG13|Tesla P40|123.0162|224|10780.7505|64|0.5203|0.14|
|VGG16|Tesla P40|128.0799|224|14749.7505|64|0.6064|0.14|
|ResNet18|Tesla P40|10.6636|224|1741.4189|64|0.1191|0.14|
|ResNet34|Tesla P40|20.3035|224|3511.4004|64|0.1947|0.14|
|ResNet50|Tesla P40|22.4385|224|3950.9077|64|0.3838|0.14|
|VGG19|Tesla P40|133.1436|224|18718.7505|64|0.6892|0.14|
|ResNet101|Tesla P40|40.5509|224|7515.5620|64|0.6041|0.14|
|ResNet152|Tesla P40|55.4698|224|11085.9585|64|0.8460|0.14|
|DenseNet121|Tesla P40|6.6415|224|2776.6565|64|0.4409|0.14|
|AlexNet|NVIDIA GeForce GTX 1060 6GB|54.4022|224|677.2448|64|0.0841|0.12|
|VGG11|NVIDIA GeForce GTX 1060 6GB|122.8402|224|7252.7505|64|0.7012|0.12|
|VGG13|NVIDIA GeForce GTX 1060 6GB|123.0162|224|10780.7505|32|0.5636|0.12|
|VGG11|NVIDIA GeForce GTX 1060 6GB|122.8402|224|7252.7505|32|0.3646|0.12|
|VGG16|NVIDIA GeForce GTX 1060 6GB|128.0799|224|14749.7505|32|0.6764|0.12|
|VGG19|NVIDIA GeForce GTX 1060 6GB|133.1436|224|18718.7505|32|0.7806|0.12|
|VGG11BN|NVIDIA GeForce GTX 1060 6GB|122.8455|224|7288.1606|32|0.4105|0.12|
|ResNet18|NVIDIA GeForce GTX 1060 6GB|10.6636|224|1741.4189|18|0.0934|0.12|
|ResNet18|NVIDIA GeForce GTX 1060 6GB|10.6636|224|1741.4189|64|0.2519|0.12|
|ResNet34|NVIDIA GeForce GTX 1060 6GB|20.3035|224|3511.4004|64|0.4191|0.12|
|ResNet50|NVIDIA GeForce GTX 1060 6GB|22.4385|224|3950.9077|32|0.4375|0.12|
|ResNet101|NVIDIA GeForce GTX 1060 6GB|40.5509|224|7515.5620|32|0.7092|0.12|
|DenseNet161|Tesla P40|25.2667|224|7508.1824|64|0.9266|0.14|
|DenseNet201|Tesla P40|17.2731|224|4208.6470|64|0.6715|0.14|
|ResNet18|NVIDIA GeForce RTX 3070 Ti|10.6636|32|35.5439|64|0.0182|0.14|
|ResNet50|NVIDIA GeForce RTX 3070 Ti|22.4385|32|80.6499|64|0.0423|0.14|
|ResNet101|NVIDIA GeForce RTX 3070 Ti|40.5509|32|153.3979|64|0.0790|0.14|
|ResNet18|NVIDIA GeForce GTX 1650|10.6636|32|35.5439|64|0.0390|0.14|
|ResNet50|NVIDIA GeForce GTX 1650|22.4385|32|80.6499|64|0.1269|0.14|
|MobileNetV2|Tesla P40|2.1331|224|317.5357|64|0.1995|0.14|
|ResNeXt50|Tesla P40|21.9349|224|4101.3291|64|0.5351|0.14|
|ResNeXt101_32|Tesla P40|82.7435|224|15800.6055|64|1.5538|0.14|
|WideResNet50|Tesla P40|63.7576|224|10936.8291|64|0.6456|0.14|
|MNASNet|Tesla P40|2.9708|224|324.6611|64|0.1877|0.14|
|ResNet18|NVIDIA GeForce MX450|10.6636|32|35.5439|64|0.0694|0.11|
|ResNet18|NVIDIA GeForce RTX 2060|10.6636|32|35.5439|64|0.0241|0.11|
|InceptionV3|Tesla P40|23.2424|299|5494.2137|64|0.6381|0.14|
|GoogleNet|Tesla P40|9.4992|224|1449.6922|64|0.1737|0.14|
|SqueezeNet1_0|Tesla P40|0.7062|224|699.2946|64|0.1247|0.14|
|SqueezeNet1_1|Tesla P40|0.6939|224|251.2845|64|0.0731|0.14|
|MobileNetV3_l|Tesla P40|4.0196|224|226.9421|64|0.1548|0.14|
|MobileNetV3_s|Tesla P40|1.4573|224|59.9614|64|0.0614|0.14|
|EfficientNet_B7|Tesla P40|60.8564|224|5135.0641|32|0.9634|0.14|
|ResNet18|NVIDIA GeForce MX330|10.6636|32|35.5439|64|0.1112|0.14|
|ResNet50|NVIDIA GeForce MX330|22.4385|32|80.6499|64|0.2850|0.14|
|ResNet18|GeForce RTX 3050 Laptop GPU|10.6636|32|35.5439|64|0.0183|0.14|
|EfficientNet_B0|Tesla P40|3.8341|224|401.1205|64|0.2787|0.14|
|ResNet50|Tesla P40|22.4385|224|3950.9077|64|0.3815|0.14|
|EfficientNet_B1|Tesla P40|6.2237|224|590.0129|64|0.3917|0.14|
|EfficientNet_B2|Tesla P40|7.3577|224|677.4029|64|0.4158|0.14|
|EfficientNet_B3|Tesla P40|10.2154|224|983.3119|64|0.5467|0.14|
|EfficientNet_B4|Tesla P40|16.7528|224|1519.9863|64|0.7364|0.14|
|EfficientNet_B5|Tesla P40|27.0474|224|2366.1147|64|1.0182|0.14|
|EfficientNet_B6|Tesla P40|38.8706|224|3356.8463|32|0.7155|0.14|
|ConvNeXt_t|Tesla P40|26.5387|224|4262.6162|64|1.9312|0.14|
|ConvNeXt_s|Tesla P40|47.1710|224|8301.3838|64|2.6355|0.14|
|ConvNeXt_b|Tesla P40|83.5197|224|14670.0117|64|3.9135|0.14|
|SwinV2_b|Tesla P40|82.8897|224|16267.2197|64|1.7707|0.14|
|ConvNeXt_l|Tesla P40|187.1545|224|32809.5176|32|3.5616|0.14|
|EfficientNetV2_s|Tesla P40|19.2550|224|2777.8542|64|0.4861|0.14|
|EfficientNetV2_m|Tesla P40|50.4219|224|5211.5523|64|0.8486|0.14|
|EfficientNetV2_l|Tesla P40|111.8155|224|11837.2903|32|0.8739|0.14|
|ResNeXt101_64|Tesla P40|77.6546|224|14891.0430|64|1.5880|0.14|
|Swin_s|Tesla P40|46.5822|224|8361.6372|64|0.9338|0.14|
|Swin_b|Tesla P40|82.7346|224|14750.3496|64|1.3730|0.14|
|Swin_t|Tesla P40|26.2518|224|4299.6138|64|0.5692|0.14|
|SwinV2_s|Tesla P40|46.7073|224|9252.4490|64|1.2432|0.14|
|VGG11BN|Tesla P40|122.8455|224|7288.1606|64|0.0111|0.14|
|VGG13BN|Tesla P40|123.0218|224|10839.1294|64|0.0142|0.14|
|VGG16BN|Tesla P40|128.0880|224|14814.3501|64|0.0175|0.14|
|VGG19BN|Tesla P40|133.1541|224|18789.5708|64|0.0199|0.14|
|AlexNet|NVIDIA GeForce GTX 1080 Ti|54.4022|224|677.2448|64|0.0350|0.12|
|ResNet18|NVIDIA GeForce GTX 1080 Ti|10.6636|224|1741.4189|64|0.0999|0.12|
|ResNet34|NVIDIA GeForce GTX 1080 Ti|20.3035|224|3511.4004|64|0.1712|0.12|
|ResNet50|NVIDIA GeForce GTX 1080 Ti|22.4385|224|3950.9077|64|0.3556|0.12|
|ResNet101|NVIDIA GeForce GTX 1080 Ti|40.5509|224|7515.5620|64|0.5682|0.12|
|DenseNet121|NVIDIA GeForce GTX 1080 Ti|6.6415|224|2776.6565|64|0.4144|0.12|
|ResNet152|NVIDIA GeForce GTX 1080 Ti|55.4698|224|11085.9585|32|0.4194|0.12|
|DenseNet169|NVIDIA GeForce GTX 1080 Ti|11.9220|224|3293.4116|32|0.2892|0.12|
|DenseNet161|NVIDIA GeForce GTX 1080 Ti|25.2667|224|7508.1824|32|0.4956|0.12|
|VGG11|NVIDIA GeForce GTX 1080 Ti|122.8402|224|7252.7505|32|0.1491|0.12|
|DenseNet201|NVIDIA GeForce GTX 1080 Ti|17.2731|224|4208.6470|32|0.3559|0.12|
|VGG16|NVIDIA GeForce GTX 1080 Ti|128.0799|224|14749.7505|32|0.2767|0.12|
|VGG19|NVIDIA GeForce GTX 1080 Ti|133.1436|224|18718.7505|64|0.6223|0.12|
|VGG16|NVIDIA GeForce GTX 1080 Ti|128.0799|224|14749.7505|64|0.5543|0.12|
|EfficientNet_B0|NVIDIA GeForce GTX 1080 Ti|3.8341|224|401.1205|64|0.2585|0.12|
|EfficientNet_B1|NVIDIA GeForce GTX 1080 Ti|6.2237|224|590.0129|64|0.3904|0.12|
|EfficientNet_B2|NVIDIA GeForce GTX 1080 Ti|7.3577|224|677.4029|64|0.4041|0.12|
|EfficientNet_B3|NVIDIA GeForce GTX 1080 Ti|10.2154|224|983.3119|32|0.2720|0.12|
|EfficientNet_B4|NVIDIA GeForce GTX 1080 Ti|16.7528|224|1519.9863|32|0.3678|0.12|
|EfficientNet_B5|NVIDIA GeForce GTX 1080 Ti|27.0474|224|2366.1147|32|0.5134|0.12|
|EfficientNet_B6|NVIDIA GeForce GTX 1080 Ti|38.8706|224|3356.8463|16|0.3606|0.12|
|EfficientNet_B7|NVIDIA GeForce GTX 1080 Ti|60.8564|224|5135.0641|16|0.5039|0.12|
|VGG11BN|NVIDIA GeForce GTX 1080 Ti|122.8455|224|7288.1606|64|0.3091|0.12|
|VGG13BN|NVIDIA GeForce GTX 1080 Ti|123.0218|224|10839.1294|64|0.5279|0.12|
|VGG19BN|NVIDIA GeForce GTX 1080 Ti|133.1541|224|18789.5708|32|0.3521|0.12|
|VGG16BN|NVIDIA GeForce GTX 1080 Ti|128.0880|224|14814.3501|64|0.6174|0.12|
|GoogleNet|NVIDIA GeForce GTX 1080 Ti|9.4992|224|1449.6922|64|0.1598|0.12|
|InceptionV3|NVIDIA GeForce GTX 1080 Ti|23.2424|299|5494.2137|64|0.5043|0.12|
|SqueezeNet1_1|NVIDIA GeForce GTX 1080 Ti|0.6939|224|251.2845|64|0.0655|0.12|
|SqueezeNet1_0|NVIDIA GeForce GTX 1080 Ti|0.7062|224|699.2946|64|0.1164|0.12|
|MobileNetV3_s|NVIDIA GeForce GTX 1080 Ti|1.4573|224|59.9614|64|0.0617|0.12|
|MobileNetV3_l|NVIDIA GeForce GTX 1080 Ti|4.0196|224|226.9421|64|0.1533|0.12|
|ShuffleNetV2|NVIDIA GeForce GTX 1080 Ti|1.2053|224|146.5275|64|0.0999|0.12|
|MobileNetV2|NVIDIA GeForce GTX 1080 Ti|2.1331|224|317.5357|64|0.1874|0.12|