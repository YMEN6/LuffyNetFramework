# README

### ARCH_SEARCH说明

##### 架构搜索流程

1. 训练超级网络（权值参数W与采样概率参数θ）；
2. 从训练后的超级网络中构建目标网络，并使用测试集对其进行初步验证；
3. 使用训练集与验证集对目标网络进行重新训练（权值参数W）；
4. 对重新训练后的目标网络进行验证；

##### 参数说明

- ```-t, --train```: 训练模式，可配合train_mode进行三种训练模式，当不选择训练模式时，默认进入两种验证模式
  - 训练模式：
  - ```-t --target super_net```：训练超网的权重w和择路参数θ
  - ```-t --target final_net```：训练从超网中构建的网络架构
  - 校验：
  - ```--target super_net```：从训练后的超网中构建网络架构，并对其进行初步校验
  - ```--target final_net```：校验已经重新训练过后的网络架构
- ```--net_config```：构造超网的配置名，目前提供resnet18与resnet18_100，前者适配CIFAR10数据集，后者适配CIFAR100数据集，不传入时默认使用resnet18_100
- ```--save_path```：本次实验保存的文件路径，请确保在一次完整实验中，该路径不会发生修改
- ```--data_path```：本次实验中，数据集所在的路径，请确保在一次完整实验中，该路径不会发生修改
- ```--dataset```: 所使用的数据集，目前支持CIFAR10和CIFAR100
- ```--check_point```：存放check point的最外层目录的路径，比如完整的check point路径为 ExpFirst/SuperNet/ck.pth， 则应指定```--check_point ExpFirst```，当不传入时，默认使用```--save_path```下的路径
- ```--final_path```：存放final net的完整路径（区分上面的check point），当不传入时，默认使用```---save_path```充当最外层目录来获取final net的存放路径
- `--device`：搜索所使用的处理器，目前仅支持单卡模式或CPU模式
- `--constrain`：指定内存约束，单位为MB，不传入时默认使用配置文件nas.ini中的参数

##### 其他说明

- 在多卡设备上若出现使用问题，可通过环境变量`CUDA_VISIBLE_DEVICES`解决问题
- 框架会自动读取checkpoint，发生中断后，会自动读取上次记录继续进行

### Demo

```python
# 首先对超网进行训练，目标是更新超网的参数w和θ
python arch_search.py --save_path experiment_path -t --data_path dataset_path --dataset cifar100 --target super_net

# 对采样子网进行初步校验
python arch_search.py --save_path experiment_path --data_path dataset_path --dataset cifar100 --target super_net

# 对采样子网进行重新训练
python arch_search.py --save_path experiment_path -t --data_path dataset_path --dataset cifar100 --target final_net

# 对重新训练后的子网进行校验
python arch_search.py --save_path experiment_path --data_path dataset_path --dataset cifar100 --target final_net
```

