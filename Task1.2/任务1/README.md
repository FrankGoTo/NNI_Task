实验流程：
    1.将训练代码中所需要调整的参数及其取值范围定义在search_space.jason文件中。
    2.在训练代码中导入search_space.jason每次更新的参数，并将每个epoch的metric结果通过接口传入nni。
    3.配置config.yml文件。
      设置多个GPU，同时允许在同一个GPU上同时运行多个不同的Trial，提升效率。
      配置assessor，当metric较差时，提前终止任务，提升效率。
    4.在命令行启动nni

search_space.jason中，超参的选择：
    初次实验时，将骨干"model"的待选类型设置为：
    'resnet18', 'resnet50', 'vgg16', 'vgg16_bn', 'densenet121', 'squeezenet1_1',
    'shufflenet_v2_x1_0', 'mobilenet_v2', 'resnext50_32x4d', 'mnasnet1_0' 
    由于cifar10数据集数量相对image net较少，且种类仅为10类。
    因此在第一次进行实验的时候，VGG、resnet50等大型网络，收敛速度较慢，测试效果并不好。
    所以将骨干网待选种类缩小为："resnet18","shufflenet_v2_x1_0", "mobilenet_v2"。
    其余参数如batch size,lr,optimizer等，都选取常见值。
    
