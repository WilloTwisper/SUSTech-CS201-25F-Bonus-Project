import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")  # 忽略"binary_closing"函数过时警告

# ================= 配置区域 (Configuration V3) =================
PRED_DIR = r"./Project_Data/Predictions"
GT_DIR = r"./Project_Data/GroundTruth"
SAVE_DIR = r"Project_Data/Report_Assets_V3"  # 保存到 V3 文件夹

# 你的目标层 ID
TARGET_CLASS_ID = 6

# 【关键参数】拟合阶数 (Polynomial Degree)
# 2 代表抛物线 (y = ax^2 + bx + c)，最适合眼底弧线。
# 如果眼球很波浪，可以改成 3。但通常 2 就够了。
POLY_DEGREE = 2

# 【关键参数】补全线条的粗细
# 你希望补出来的线大概多少像素宽？根据你的原图，3或5比较合适。
LINE_THICKNESS = 4
# ===========================================================

os.makedirs(SAVE_DIR, exist_ok=True)


def fit_curve_and_bridge(binary_mask):
    """
    [V3 Algorithm]
    使用多项式拟合 (Polynomial Fitting) 来预测并连接断裂的解剖结构。
    """
    h, w = binary_mask.shape

    # 1. 提取所有前景像素的坐标
    # y_coords, x_coords = (row, col)
    y_coords, x_coords = np.where(binary_mask > 0)

    # 如果检测到的点太少，没法拟合，直接返回原图
    if len(x_coords) < 10:
        return binary_mask, binary_mask, False

    # 2. 【核心数学】最小二乘法多项式拟合
    # 我们拟合 y = f(x)
    try:
        # coefficients 包含 [a, b, c] 对应 ax^2 + bx + c
        coeffs = np.polyfit(x_coords, y_coords, POLY_DEGREE)
        poly_func = np.poly1d(coeffs)
    except:
        return binary_mask, binary_mask, False

    # 3. 生成拟合曲线 (The "Ideal" Curve)
    fitted_mask = np.zeros_like(binary_mask)

    # 在 x 轴范围内生成点
    # 为了防止只在有像素的地方生成，我们应该覆盖所有碎片的 x 范围
    x_min, x_max = np.min(x_coords), np.max(x_coords)

    # 生成一系列密集的 x 点
    x_range = np.linspace(x_min, x_max, num=(x_max - x_min) * 2).astype(int)
    y_range = poly_func(x_range).astype(int)

    # 4. 绘制曲线 (Rasterization)
    # 过滤掉跑出图像边界的点
    valid_idx = (y_range >= 0) & (y_range < h) & (x_range >= 0) & (x_range < w)

    # 简单的画点 (1px)
    # fitted_mask[y_range[valid_idx], x_range[valid_idx]] = 1

    # 更好的画法：用 cv2.polylines 画一条有厚度的线
    pts = np.column_stack((x_range[valid_idx], y_range[valid_idx]))
    cv2.polylines(fitted_mask, [pts], isClosed=False, color=1, thickness=LINE_THICKNESS)

    # 5. 融合 (Fusion)
    # 策略：保留原始预测 + 加上拟合曲线填补空缺
    # 使用逻辑或 (OR)
    merged_mask = np.logical_or(binary_mask, fitted_mask).astype(np.uint8)

    return merged_mask, fitted_mask, True


def refine_mask_v3(merged_mask):
    """
    后处理：只保留融合后的最大连通块
    """
    label_img, num_labels = measure.label(merged_mask, return_num=True, connectivity=2)

    if num_labels <= 1:
        return merged_mask

    properties = measure.regionprops(label_img)
    if not properties: return merged_mask

    max_prop = max(properties, key=lambda x: x.area)
    refined = np.zeros_like(merged_mask)
    refined[label_img == max_prop.label] = 1

    return refined


def dice_score(pred, gt):
    smooth = 1e-5
    intersection = np.sum(pred * gt)
    return (2. * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)


def main():
    pred_files = sorted(glob(os.path.join(PRED_DIR, "*.png")))

    print(f"开始 V3 曲线拟合分析 (Class: {TARGET_CLASS_ID}, Degree: {POLY_DEGREE})...")
    print(f"结果图片将保存到: {SAVE_DIR}")

    metrics = {'orig_dice': [], 'refi_dice': [], 'orig_b0': []}

    saved_count = 0
    MAX_SAVE = 10

    for p_path in tqdm(pred_files):
        filename = os.path.basename(p_path)
        g_path = os.path.join(GT_DIR, filename)
        if not os.path.exists(g_path): continue

        pred_full = cv2.imread(p_path, 0)
        gt_full = cv2.imread(g_path, 0)
        pred_bin = (pred_full == TARGET_CLASS_ID).astype(np.uint8)
        gt_bin = (gt_full == TARGET_CLASS_ID).astype(np.uint8)

        if np.sum(gt_bin) == 0 and np.sum(pred_bin) == 0: continue

        b0_raw = measure.label(pred_bin, return_num=True, connectivity=2)[1]

        # === V4 核心 ===
        # 1. 拟合曲线并合并
        merged_mask, fitted_curve_only, success = fit_curve_and_bridge(pred_bin)

        # 2. 提取最大连通块
        pred_refined = refine_mask_v3(merged_mask)

        # 指标
        d_orig = dice_score(pred_bin, gt_bin)
        d_refi = dice_score(pred_refined, gt_bin)

        metrics['orig_dice'].append(d_orig)
        metrics['refi_dice'].append(d_refi)
        metrics['orig_b0'].append(b0_raw)

        # 可视化：找那种断开的，且修复后效果好的
        if b0_raw > 1 and success and saved_count < MAX_SAVE:
            saved_count += 1

            plt.figure(figsize=(16, 5))

            # 1. GT
            plt.subplot(1, 4, 1)
            plt.title("Ground Truth", fontsize=10)
            plt.imshow(gt_bin, cmap='gray')
            plt.axis('off')

            # 2. Raw
            plt.subplot(1, 4, 2)
            plt.title(f"Original (Broken)\nDice: {d_orig:.3f} | $b_0$: {b0_raw}", fontsize=10)
            plt.imshow(pred_bin, cmap='gray')
            plt.axis('off')

            # 3. Mathematical Fitting
            plt.subplot(1, 4, 3)
            plt.title(f"Polynomial Fitting ($y=ax^2+bx+c$)\nGlobal Trend Extracted", fontsize=10)
            # 把拟合的线叠在原图上显示，效果更直观
            combined_vis = pred_bin.copy().astype(float)
            combined_vis[fitted_curve_only == 1] = 0.5  # 灰色显示拟合线
            plt.imshow(combined_vis, cmap='gray')
            plt.axis('off')

            # 4. Final
            plt.subplot(1, 4, 4)
            plt.title(f"V3 Reconstruction\nDice: {d_refi:.3f} (Bridged!)", fontsize=10)
            plt.imshow(pred_refined, cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, f"V3_Fit_{filename}"))
            plt.close()

    # ================= 统计 =================
    print("\n" + "=" * 40)
    print(" V3 (CURVE FITTING) RESULTS SUMMARY ")
    print("=" * 40)
    print("Dice Coefficient Score:")
    print(f"   - Original Mean Dice: {np.mean(metrics['orig_dice']):.4f}")
    print(f"   - Refined  Mean Dice: {np.mean(metrics['refi_dice']):.4f}")

    imp = np.mean(metrics['refi_dice']) - np.mean(metrics['orig_dice'])
    print(f"   - Improvement: {imp:+.4f}")
    if imp > 0:
        print("   ✅ SUCCESS: High-precision reconstruction!")
    else:
        print("   ⚠️ Check if thickness matches.")
    print("=" * 40)


if __name__ == "__main__":
    main()