# sonar_baseline_with_tensorflow
## 项目说明
本项目用于[和鲸平台比赛](https://www.kesci.com/home/competition/5e535a612537a0002ca864ac)，实践了用户 [Pumpkin](https://www.kesci.com/home/user/profile/5da7e869048089002c7d2f58) 分享的 [baseline](https://www.kesci.com/home/project/5e6331644b7a30002c98895e)

## 参考项目链接
1. [Protocol Buffer](https://github.com/protocolbuffers/protobuf.git) 
2. [Tensorflow 开源模型](https://github.com/tensorflow/models.git)
3. [TF-slim](https://github.com/google-research/tf-slim.git)

## 环境
- Tensorflow 1.15.2
- lxml 4.5.0
- Pillow 7.0.0
- matplotlib 3.2.0
- PyYaml 5.3

## 使用方法
1. 运行 `tfrecord_generator.py`, 采用 `-path` 参数传入大赛数据集的**压缩包** 
2. 运行 `model_train.py`, 采用 `-path` 参数传入预训练模型的文件夹地址