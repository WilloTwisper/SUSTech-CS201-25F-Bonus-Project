import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.morphology import binary_closing, disk
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")  # 忽略"binary_closing"函数过时警告

# ================= 配置区域 (Configuration V2) =================
PRED_DIR = r"./Project_Data/Predictions"
GT_DIR = r"./Project_Data/GroundTruth"
SAVE_DIR = r"Project_Data/Report_Assets_V2"  # 保存到 V3 文件夹

# 你的目标层 ID (根据审计结果，这里填 6)
TARGET_CLASS_ID = 6

# 【关键参数】闭运算的半径 (Closing Radius)
# 这个值越大，能连上的缝隙就越宽。
# 建议从 10 开始尝试。如果还有断裂，试着改大到 15 或 20。
CLOSING_RADIUS = 10
# ===========================================================

os.makedirs(SAVE_DIR, exist_ok=True)


def apply_morphology_and_topology(binary_mask):
    """
    [V2 Upgrade]
    先进行形态学闭运算，再计算拓扑特征。
    目的：修复断裂 (Fracture Repair)
    """
    # 1. 定义结构元素 (Structuring Element)
    # disk(r) 生成一个圆形矩阵，半径为 r
    selem = disk(CLOSING_RADIUS)

    # 2. 执行闭运算: 先膨胀再腐蚀
    # 这会连接距离小于 2*r 的断裂处，并填补孔洞
    closed_mask = binary_closing(binary_mask, selem)

    # 3. 基于修复后的图计算连通性
    # label_img: 连通域标记图
    label_img, num_labels = measure.label(closed_mask, return_num=True, connectivity=2)

    # 计算欧拉数 (基于修复后的图)
    euler_val = measure.euler_number(closed_mask, connectivity=2)

    return num_labels, euler_val, label_img, closed_mask


def refine_mask_v2(closed_mask, label_img, num_labels):
    """
    [V2 Algorithm]
    基于修复后的图进行最大连通分量提取
    """
    if num_labels <= 1:
        return closed_mask, False  # 本身就是连通的，直接返回闭运算结果

    # 计算每个连通分量的面积
    properties = measure.regionprops(label_img)
    if not properties:
        return closed_mask, False

    # 找到最大的那个块
    max_prop = max(properties, key=lambda x: x.area)

    # 只保留这个最大的块
    refined = np.zeros_like(closed_mask)
    refined[label_img == max_prop.label] = 1

    return refined, True


def dice_score(pred, gt):
    smooth = 1e-5
    intersection = np.sum(pred * gt)
    return (2. * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)


def main():
    pred_files = sorted(glob(os.path.join(PRED_DIR, "*.png")))

    print(f"开始 V2 核心分析 (Target Class: {TARGET_CLASS_ID}, Radius: {CLOSING_RADIUS})...")
    print(f"结果图片将保存到: {SAVE_DIR}")

    metrics = {
        'orig_dice': [],
        'refi_dice': [],
        'orig_b0': [],
        'refi_b0': []  # 这里记录的是闭运算后的 b0
    }

    saved_images_count = 0
    MAX_SAVE = 10

    for p_path in tqdm(pred_files):
        filename = os.path.basename(p_path)
        g_path = os.path.join(GT_DIR, filename)

        if not os.path.exists(g_path):
            continue

        # 1. 读取数据
        pred_full = cv2.imread(p_path, 0)
        gt_full = cv2.imread(g_path, 0)

        # 2. 提取二值图
        pred_bin = (pred_full == TARGET_CLASS_ID).astype(np.uint8)
        gt_bin = (gt_full == TARGET_CLASS_ID).astype(np.uint8)

        if np.sum(gt_bin) == 0 and np.sum(pred_bin) == 0:
            continue

        # 3. [V2] 形态学 + 拓扑计算
        # 注意：这里我们算一下原始的 b0 只是为了对比记录
        b0_raw = measure.label(pred_bin, return_num=True, connectivity=2)[1]

        # 核心操作
        b0_closed, chi_closed, label_img, closed_mask = apply_morphology_and_topology(pred_bin)

        # 4. [V2] 修正
        pred_refined, was_refined = refine_mask_v2(closed_mask, label_img, b0_closed)

        # 5. 计算指标
        d_orig = dice_score(pred_bin, gt_bin)
        d_refi = dice_score(pred_refined, gt_bin)

        metrics['orig_dice'].append(d_orig)
        metrics['refi_dice'].append(d_refi)
        metrics['orig_b0'].append(b0_raw)
        metrics['refi_b0'].append(1)  # 修正后强制为1 (最大连通块)

        # 6. 可视化 (重点关注 V1 失败的案例)
        # 条件：原始图断裂了(b0_raw > 1) 且 闭运算成功把它们连起来了(b0_closed < b0_raw)
        # 或者 Dice 提升了
        is_interesting = (b0_raw > 1) and (d_refi > d_orig)

        if is_interesting and saved_images_count < MAX_SAVE:
            saved_images_count += 1

            plt.figure(figsize=(16, 5))

            # 1. GT
            plt.subplot(1, 4, 1)
            plt.title("Ground Truth", fontsize=10)
            plt.imshow(gt_bin, cmap='gray')
            plt.axis('off')

            # 2. Original (Fractured)
            plt.subplot(1, 4, 2)
            plt.title(f"Original Prediction\nDice: {d_orig:.3f} | $b_0$: {b0_raw} (Fractured)", fontsize=10)
            plt.imshow(pred_bin, cmap='gray')
            plt.axis('off')

            # 3. Closed & Labeled (Bridged)
            plt.subplot(1, 4, 3)
            plt.title(f"Morphological Closing (r={CLOSING_RADIUS})\nGap Bridged! $b_0$ becomes {b0_closed}",
                      fontsize=10)
            plt.imshow(label_img, cmap='nipy_spectral')
            plt.axis('off')

            # 4. Final Result
            plt.subplot(1, 4, 4)
            plt.title(f"Final Refined V2\nDice: {d_refi:.3f} (Improved!)", fontsize=10)
            plt.imshow(pred_refined, cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, f"V2_Result_{filename}"))
            plt.close()

    # ================= 统计输出 =================
    print("\n" + "=" * 40)
    print(" V2 EXPERIMENTAL RESULTS SUMMARY ")
    print("=" * 40)
    print(f"Algorithm: Morphological Closing (r={CLOSING_RADIUS}) + Topo Refinement")
    print(f"Total Samples: {len(metrics['orig_dice'])}")
    print("-" * 40)
    print("Dice Coefficient Score:")
    print(f"   - Original Mean Dice: {np.mean(metrics['orig_dice']):.4f}")
    print(f"   - Refined  Mean Dice: {np.mean(metrics['refi_dice']):.4f}")

    imp = np.mean(metrics['refi_dice']) - np.mean(metrics['orig_dice'])
    print(f"   - Improvement: {imp:+.4f}")
    if imp > 0:
        print("   ✅ SUCCESS: Gaps bridged, accuracy improved!")
    else:
        print("   ⚠️ Note: Radius might be too small/large.")

    print("=" * 40)


if __name__ == "__main__":
    main()