import datetime
import json
import matplotlib.pyplot as plt

def write_training_report(
    save_path,
    dataset_name,
    model_name1,
    model_name2,
    best_fid1,
    best_fid1_epoch,
    best_fid2,
    best_fid2_epoch,
    data_config,
    exp_config,
    dit_config,
    meanflow_config,
    alpha_config
):
    """
    生成训练报告,包含基本信息、超参数、两个模型的最佳FID及其epoch。
    """
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = []
    lines.append(f"# 训练报告\n")
    lines.append(f"生成时间: {now}\n")
    lines.append(f"数据集: {dataset_name}\n")
    lines.append(f"模型1: {model_name1}")
    lines.append(f"模型2: {model_name2}\n")
    lines.append(f"## 最佳FID\n")
    lines.append(f"- {model_name1} 最佳FID: {best_fid1:.2f} (Epoch {best_fid1_epoch})")
    lines.append(f"- {model_name2} 最佳FID: {best_fid2:.2f} (Epoch {best_fid2_epoch})\n")
    lines.append(f"## 超参数配置\n")
    lines.append(f"### DATA_CONFIG\n{json.dumps(data_config, indent=2, ensure_ascii=False)}\n")
    lines.append(f"### EXPERIMENT_CONFIG\n{json.dumps(exp_config, indent=2, ensure_ascii=False)}\n")
    lines.append(f"### DIT_CONFIG\n{json.dumps(dit_config, indent=2, ensure_ascii=False)}\n")
    lines.append(f"### MEANFLOW_CONFIG\n{json.dumps(meanflow_config, indent=2, ensure_ascii=False)}\n")
    lines.append(f"### ALPHA_CONFIG\n{json.dumps(alpha_config, indent=2, ensure_ascii=False)}\n")
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def plot_training_curves(losses, add_losses, fids, add_fids, alphas, lrs, save_path='Compare_Reports/images/training_curves.png'):
    """
    绘制训练过程中的loss、FID、alpha、lr曲线,保存为2x2图。
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # 1. Loss 曲线
    axs[0, 0].plot(losses, label='MeanFlow Loss')
    axs[0, 0].plot(add_losses, label='AdditiveMeanFlow Loss')
    axs[0, 0].set_title('Loss Curve')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    # 2. FID 曲线
    axs[0, 1].plot(fids, label='MeanFlow FID')
    axs[0, 1].plot(add_fids, label='AdditiveMeanFlow FID')
    axs[0, 1].set_title('FID Curve')
    axs[0, 1].set_xlabel('Eval Step')
    axs[0, 1].set_ylabel('FID')
    axs[0, 1].legend()
    # 3. Alpha 曲线
    axs[1, 0].plot(alphas, label='Alpha')
    axs[1, 0].set_title('Alpha Curve')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Alpha')
    axs[1, 0].legend()
    # 4. LR 曲线
    axs[1, 1].plot(lrs, label='Learning Rate')
    axs[1, 1].set_title('Learning Rate Curve')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('LR')
    axs[1, 1].legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
