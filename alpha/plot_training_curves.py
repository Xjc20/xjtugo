"""
绘制训练曲线脚本

功能：
1. 读取training_log.txt文件
2. 解析Loss和PolicyEntropy数据
3. 将每一行视为一轮训练（不考虑Iteration数字）
4. 绘制训练轮数与Loss和PolicyEntropy的折线图
5. 保存到alpha目录下
"""

import os
import re
import matplotlib.pyplot as plt

def plot_training_curves():
    # 文件路径
    log_file = os.path.join(os.path.dirname(__file__), "training_log.txt")
    output_file = os.path.join(os.path.dirname(__file__), "training_curves_new.png")
    
    # 检查文件是否存在
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found")
        return
    
    # 读取数据
    iterations = []
    losses = []
    entropies = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析每一行数据
    for i, line in enumerate(lines, 1):  # 从1开始计数，第一行就是第一轮
        line = line.strip()
        if not line:
            continue
        
        # 使用正则表达式提取Loss和PolicyEntropy
        loss_match = re.search(r'Loss=([\d.]+)', line)
        entropy_match = re.search(r'PolicyEntropy=([\d.]+)', line)
        
        if loss_match and entropy_match:
            loss = float(loss_match.group(1))
            entropy = float(entropy_match.group(1))
            
            iterations.append(i)
            losses.append(loss)
            entropies.append(entropy)
    
    if not iterations:
        print("No valid data found in the log file")
        return
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 绘制Loss曲线
    ax1.plot(iterations, losses, 'b-', linewidth=2, label='Loss')
    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 绘制PolicyEntropy曲线
    ax2.plot(iterations, entropies, 'r-', linewidth=2, label='Policy Entropy')
    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('Policy Entropy')
    ax2.set_title('Policy Entropy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {output_file}")
    
    # 显示统计信息
    print(f"\nTotal training rounds: {len(iterations)}")
    print(f"Initial Loss: {losses[0]:.4f}, Final Loss: {losses[-1]:.4f}")
    print(f"Initial Entropy: {entropies[0]:.4f}, Final Entropy: {entropies[-1]:.4f}")
    
    plt.close()

if __name__ == "__main__":
    plot_training_curves()
