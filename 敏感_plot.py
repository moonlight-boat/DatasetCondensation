import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

# 读取 CSV 文件
file_path = 'result/class_acc_test_results_ipc_敏感.csv'  # 替换成你的 CSV 文件路径
df = pd.read_csv(file_path)

# 提取各个类的测试准确率列和 IPC 列
class_acc_columns = [f'Class_{i+1}_Acc' for i in range(10)]  # Class_1_Acc 到 Class_10_Acc

# 定义类别名称 (CIFAR-10 类别名)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 定义颜色循环
colors = plt.get_cmap('tab10').colors
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P', 'H']  # 不同形状的标记

# 设置颜色和标记样式
plt.figure(figsize=(10, 6))
plt.rc('axes', prop_cycle=(cycler('color', colors)))

# 遍历每个类的测试准确率列
for idx, class_acc in enumerate(class_acc_columns):
    # 提取测试准确率和对应的 IPC 值 (测试准确率和 IPC 值交替出现)
    acc_values = df[class_acc][::2].values  # 每隔一行提取测试准确率
    ipc_values = df[class_acc][1::2].values  # 提取测试准确率下面的 IPC 值

    # 绘制类的测试准确率 vs IPC，使用不同的颜色和标记
    plt.plot(ipc_values, acc_values, marker=markers[idx], label=class_names[idx], markersize=8, linewidth=2)

# 设置图表标题和标签
plt.xlabel('IPC', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.title('Test Accuracy vs IPC for Each Class', fontsize=16)

# 设置图例位置
plt.legend(title='Class', loc='best', fontsize=10)

# 显示网格
plt.grid(True)

# 保存图像到文件
save_path = 'result/class_acc_vs_ipc_styled.png'  # 替换为你希望保存的路径和文件名
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# 显示图像
plt.show()
