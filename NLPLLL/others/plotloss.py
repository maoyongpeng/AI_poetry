import matplotlib.pyplot as plt


def plotloss( loss_file_path):
    # 初始化数据列表
    epochs = []
    losses = []
    cls=loss_file_path.split('_')[1].split(".")[0]
    # 读取文件并解析数据
    with open(loss_file_path,'r') as file:
        for line in file:
            parts = line.strip().split(', ')
            epoch = int(parts[0].split()[1])
            loss = float(parts[1].split()[1])
            epochs.append(epoch)
            losses.append(loss)

    # 绘制Loss曲线图
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    # 保存图像
    plt.savefig(f'loss_curve_'+cls+'.png')

    # 显示图像
    plt.show()

# loss_file_path='loss_ci.txt'
# cls=loss_file_path.split('_')[1].split(".")[0]
# print(cls)

plotloss(loss_file_path='loss_ci.txt')