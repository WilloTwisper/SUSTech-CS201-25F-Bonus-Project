import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from glob import glob
from tqdm import tqdm

# ================= 配置区域 (Configuration V5) =================
PRED_DIR = r"./Project_Data/Predictions"
GT_DIR = r"./Project_Data/GroundTruth"
SAVE_DIR = r"./Project_Data/Report_Assets_V5"  # 保存到 V5 文件夹

TARGET_CLASS_ID = 6
POLY_DEGREE = 2  # 2次多项式适合眼球弧度
# 注意：移除了 BRIDGE_THICKNESS，因为现在厚度是自动计算的！
# ================================================================

os.makedirs(SAVE_DIR, exist_ok=True)


def fit_extrapolate_adaptive_thickness(binary_mask):
    """
    [V5 Algorithm - Dual Regression]
    1. 计算距离变换 (得到局部厚度 r)
    2. 拟合位置曲线 y = f(x)
    3. 拟合厚度曲线 r = g(x)
    4. 动态绘制变宽的桥梁
    """
    h, w = binary_mask.shape

    # 1. 提取前景点坐标
    y_coords, x_coords = np.where(binary_mask > 0)

    if len(x_coords) < 10:
        return binary_mask, binary_mask, False

    # 2. 【新增】计算局部厚度 (Radius)
    # distanceTransform 计算每个像素到最近背景像素的距离
    # 对于管状结构，中心线的距离值就是半径
    dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    # 获取前景点的厚度值
    r_vals = dist_map[y_coords, x_coords]

    # 3. 双重拟合 (Dual Fitting)
    try:
        # A. 位置拟合 y = f(x)
        coeffs_y = np.polyfit(x_coords, y_coords, POLY_DEGREE)
        poly_y = np.poly1d(coeffs_y)

        # B. 【新增】厚度拟合 r = g(x)
        # 厚度变化通常较平缓，2次多项式足够描述 (中间厚两边薄)
        coeffs_r = np.polyfit(x_coords, r_vals, POLY_DEGREE)
        poly_r = np.poly1d(coeffs_r)

    except:
        return binary_mask, binary_mask, False

    # 4. 全域外推 (Extrapolation)
    x_range = np.arange(0, w).astype(int)

    # 预测位置
    y_pred = poly_y(x_range).astype(int)

    # 预测厚度 (半径)
    # 注意：需防止负值，且设定一个最小厚度(比如1px)
    r_pred = poly_r(x_range)
    r_pred = np.clip(r_pred, 1.0, 50.0)  # 限制半径在 1 到 50 之间

    # 5. 生成"动态桥梁" (Adaptive Bridge Layer)
    bridge_mask = np.zeros_like(binary_mask)

    # 由于 openCV 的 polylines 只能画等宽线
    # 我们改用 circle 逐点绘制来实现"渐变粗细"
    # 为了效率，可以每隔几个像素画一个圆，或者逐点画
    for i, x in enumerate(x_range):
        y = y_pred[i]
        r = r_pred[i]

        # 只在图像范围内画
        if 0 <= y < h:
            # radius 必须是整数，且至少为 1
            radius = int(round(r))
            # 画实心圆 (-1)
            cv2.circle(bridge_mask, (x, y), radius, 1, -1)

    # 6. 保留原貌融合
    merged_mask = np.logical_or(binary_mask, bridge_mask).astype(np.uint8)

    return merged_mask, bridge_mask, True


def refine_mask_final(merged_mask):
    """提取最大连通块 (保持不变)"""
    label_img, num_labels = measure.label(merged_mask, return_num=True, connectivity=2)

    if num_labels <= 1: return merged_mask

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

    print(f"开始 V5 双重回归分析 (Dual Regression: Position + Thickness)...")
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

        # === V5 运行 ===
        merged_mask, bridge_layer, success = fit_extrapolate_adaptive_thickness(pred_bin)
        pred_refined = refine_mask_final(merged_mask)

        # 指标
        d_orig = dice_score(pred_bin, gt_bin)
        d_refi = dice_score(pred_refined, gt_bin)

        metrics['orig_dice'].append(d_orig)
        metrics['refi_dice'].append(d_refi)
        metrics['orig_b0'].append(b0_raw)

        # 可视化：找那种断开的
        if b0_raw > 1 and success and saved_count < MAX_SAVE:
            saved_count += 1
            plt.figure(figsize=(16, 5))

            # 1. GT
            plt.subplot(1, 4, 1)
            plt.title("Ground Truth", fontsize=10)
            plt.imshow(gt_bin, cmap='gray')
            plt.axis('off')

            # 2. Original
            plt.subplot(1, 4, 2)
            plt.title(f"Original (Broken)\nDice: {d_orig:.3f}", fontsize=10)
            plt.imshow(pred_bin, cmap='gray')
            plt.axis('off')

            # 3. The Bridge (Visualized)
            plt.subplot(1, 4, 3)
            plt.title(f"Dual Regression Prediction\n(Adaptive Thickness)", fontsize=10)

            # 可视化技巧：显示桥梁的粗细变化
            vis_layer = np.zeros_like(pred_bin, dtype=float)
            vis_layer[pred_bin == 1] = 0.4  # 原图灰色
            vis_layer[bridge_layer == 1] = 1.0  # 拟合线亮白色

            plt.imshow(vis_layer, cmap='gray')
            plt.axis('off')

            # 4. Final Result
            plt.subplot(1, 4, 4)
            plt.title(f"V5 Final Result\nDice: {d_refi:.3f}", fontsize=10)
            plt.imshow(pred_refined, cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, f"V5_Adaptive_{filename}"))
            plt.close()

    # ================= 统计 =================
    print("\n" + "=" * 40)
    print(" V5 FINAL RESULTS SUMMARY ")
    print("=" * 40)
    print("Dice Coefficient Score:")
    print(f"   - Original Mean Dice: {np.mean(metrics['orig_dice']):.4f}")
    print(f"   - Refined  Mean Dice: {np.mean(metrics['refi_dice']):.4f}")

    imp = np.mean(metrics['refi_dice']) - np.mean(metrics['orig_dice'])
    print(f"   - Improvement: {imp:+.4f}")
    print("   ✅ SUCCESS: Adaptive Thickness Reconstruction Completed!")
    print("=" * 40)


if __name__ == "__main__":
    main()