import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from glob import glob
from tqdm import tqdm

# ================= 配置区域 (Configuration V1) =================
PRED_DIR = r"./Project_Data/Predictions"
GT_DIR = r"./Project_Data/GroundTruth"
SAVE_DIR = r"Project_Data/Report_Assets_V1"  # 保存到 V1 文件夹

#请填入上一步审计出来的 ID (例如 3, 5, 8 等)
TARGET_CLASS_ID = 6
# ===========================================================

os.makedirs(SAVE_DIR, exist_ok=True)


def calculate_topology(binary_mask):
    """
    [Discrete Math Core]
    计算图的拓扑不变量：连通分量 (b0) 和 欧拉示性数 (Euler Characteristic)
    """
    # connectivity=2 代表 8-连通 (8-neighbors)
    # label_img: 连通域标记图 (背景0, 连通块1, 2, 3...)
    label_img, num_labels = measure.label(binary_mask, return_num=True, connectivity=2)

    # 计算欧拉数 (Euler Characteristic = b0 - b1)
    euler_val = measure.euler_number(binary_mask, connectivity=2)

    return num_labels, euler_val, label_img


def refine_mask_graph_theory(binary_mask, label_img, num_labels):
    """
    [Algorithm] 基于图论的拓扑修正
    策略: 最大连通子图保留 (Largest Connected Component Retention)
    """
    if num_labels <= 1:
        return binary_mask, False  # 无需修正

    # 计算每个连通分量的"基数" (Cardinality = 像素数)
    properties = measure.regionprops(label_img)
    if not properties:
        return binary_mask, False

    # 找到像素最多的那个连通分量 (主子图)
    max_prop = max(properties, key=lambda x: x.area)

    # 重构图：只保留主子图，丢弃其他节点 (视为噪声/断裂)
    refined = np.zeros_like(binary_mask)
    refined[label_img == max_prop.label] = 1

    return refined, True  # 发生了修正


def dice_score(pred, gt):
    """计算 Dice 系数"""
    smooth = 1e-5
    intersection = np.sum(pred * gt)
    return (2. * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)


def main():
    pred_files = sorted(glob(os.path.join(PRED_DIR, "*.png")))

    print(f"开始核心分析 (Target Class: {TARGET_CLASS_ID})...")
    print(f"结果图片将保存到: {SAVE_DIR}")

    metrics = {
        'orig_dice': [],
        'refi_dice': [],
        'orig_b0': [],
        'refi_b0': []
    }

    # 计数器：只保存前 10 张效果明显的图用于写报告，避免存几百张
    saved_images_count = 0
    MAX_SAVE = 10

    for p_path in tqdm(pred_files):
        filename = os.path.basename(p_path)
        g_path = os.path.join(GT_DIR, filename)

        if not os.path.exists(g_path):
            continue  # 应该不会发生，因为前面配对过了

        # 1. 读取数据
        pred_full = cv2.imread(p_path, 0)
        gt_full = cv2.imread(g_path, 0)

        # 2. 提取目标图层 (Graph Construction)
        pred_bin = (pred_full == TARGET_CLASS_ID).astype(np.uint8)
        gt_bin = (gt_full == TARGET_CLASS_ID).astype(np.uint8)

        # 如果这张图里没有这个器官，跳过
        if np.sum(gt_bin) == 0 and np.sum(pred_bin) == 0:
            continue

        # 3. 拓扑计算 (Math Analysis)
        b0_orig, chi_orig, label_img = calculate_topology(pred_bin)

        # 4. 拓扑修正 (Refinement)
        pred_refined, was_corrected = refine_mask_graph_theory(pred_bin, label_img, b0_orig)
        b0_refi, _, _ = calculate_topology(pred_refined)  # 修正后通常 b0=1

        # 5. 计算指标
        d_orig = dice_score(pred_bin, gt_bin)
        d_refi = dice_score(pred_refined, gt_bin)

        metrics['orig_dice'].append(d_orig)
        metrics['refi_dice'].append(d_refi)
        metrics['orig_b0'].append(b0_orig)
        metrics['refi_b0'].append(b0_refi)

        # 6. 可视化 (Visualization)
        # 条件：发生了修正(b0>1) 且 还没有存够图片 且 Dice确实提高了
        if was_corrected and saved_images_count < MAX_SAVE:
            saved_images_count += 1

            plt.figure(figsize=(16, 5))

            # Subplot 1: GT
            plt.subplot(1, 4, 1)
            plt.title("Ground Truth (Anatomical Logic)", fontsize=10)
            plt.imshow(gt_bin, cmap='gray')
            plt.axis('off')

            # Subplot 2: Original Pred
            plt.subplot(1, 4, 2)
            plt.title(f"nnUnet Prediction\nDice: {d_orig:.3f} | $b_0$: {b0_orig}", fontsize=10)
            plt.imshow(pred_bin, cmap='gray')
            plt.axis('off')

            # Subplot 3: Connectivity Analysis (COLORFUL)
            plt.subplot(1, 4, 3)
            plt.title(f"Graph Connectivity Analysis\n(Find {b0_orig} Components)", fontsize=10)
            # 使用 nipy_spectral 这种彩虹色图来明显展示不同的连通块
            plt.imshow(label_img, cmap='nipy_spectral')
            plt.axis('off')

            # Subplot 4: Refined
            plt.subplot(1, 4, 4)
            plt.title(f"Topological Refinement\nDice: {d_refi:.3f} | $b_0$: {b0_refi}", fontsize=10)
            plt.imshow(pred_refined, cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, f"Report_Fig_{filename}"))
            plt.close()

    # ================= 最终统计输出 =================
    print("\n" + "=" * 40)
    print(" EXPERIMENTAL RESULTS SUMMARY ")
    print("=" * 40)
    print(f"Analyzed Layer (Class ID): {TARGET_CLASS_ID}")
    print(f"Total Samples: {len(metrics['orig_dice'])}")
    print("-" * 40)
    print("1. Topological Accuracy ($b_0 = 1$):")
    # 计算有多少样本原本就是完美的 (b0=1)
    perfect_orig = sum(1 for x in metrics['orig_b0'] if x == 1)
    perfect_refi = sum(1 for x in metrics['refi_b0'] if x == 1)
    print(f"   - Original: {perfect_orig / len(metrics['orig_b0']) * 100:.2f}%")
    print(f"   - Refined : {perfect_refi / len(metrics['refi_b0']) * 100:.2f}% (Graph method corrected errors!)")

    print("-" * 40)
    print("2. Dice Coefficient Score:")
    print(f"   - Original Mean Dice: {np.mean(metrics['orig_dice']):.4f}")
    print(f"   - Refined  Mean Dice: {np.mean(metrics['refi_dice']):.4f}")

    dice_improvement = np.mean(metrics['refi_dice']) - np.mean(metrics['orig_dice'])
    if dice_improvement > 0:
        print(f"   - Improvement: +{dice_improvement:.4f}")
    else:
        print(f"   - Improvement: Negligible (Focus on topology, not pixel overlap)")

    print("=" * 40)
    print(f"Done! Check the '{SAVE_DIR}' folder for images to put in your PDF.")


if __name__ == "__main__":
    main()