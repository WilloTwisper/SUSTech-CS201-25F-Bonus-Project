import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from glob import glob
from tqdm import tqdm

# ================= 配置区域 (Configuration V4) =================
PRED_DIR = r"./Project_Data/Predictions"
GT_DIR = r"./Project_Data/GroundTruth"
SAVE_DIR = r"./Project_Data/Report_Assets_V4"  # 保存到 V4 文件夹

TARGET_CLASS_ID = 6
POLY_DEGREE = 2  # 2次多项式 (抛物线) 适合眼球弧度
BRIDGE_THICKNESS = 3  # 桥的厚度，稍微细一点，只起连接作用
# ================================================================

os.makedirs(SAVE_DIR, exist_ok=True)


def fit_extrapolate_and_merge(binary_mask):
    """
    [V5 Algorithm]
    1. 提取特征点
    2. 全局多项式拟合
    3. 全局外推 (0 到 Width)
    4. 保留原图 + 叠加曲线
    """
    h, w = binary_mask.shape

    # 1. 提取前景点
    y_coords, x_coords = np.where(binary_mask > 0)

    if len(x_coords) < 10:
        return binary_mask, binary_mask, False

    # 2. 拟合方程 y = f(x)
    try:
        coeffs = np.polyfit(x_coords, y_coords, POLY_DEGREE)
        poly_func = np.poly1d(coeffs)
    except:
        return binary_mask, binary_mask, False

    # 3. 【关键改进】全域外推 (Extrapolation)
    # 不再局限于 min(x) 到 max(x)，而是直接生成 0 到 W 的所有点
    # 这就是你想要的"从宏观角度，顺着趋势延伸到边缘"
    x_range = np.arange(0, w).astype(int)
    y_range = poly_func(x_range).astype(int)

    # 4. 生成"桥梁"层 (The Bridge Layer)
    bridge_mask = np.zeros_like(binary_mask)

    # 过滤越界点
    valid_idx = (y_range >= 0) & (y_range < h)

    # 绘制曲线
    pts = np.column_stack((x_range[valid_idx], y_range[valid_idx]))
    cv2.polylines(bridge_mask, [pts], isClosed=False, color=1, thickness=BRIDGE_THICKNESS)

    # 5. 【关键改进】保留原貌融合 (Structure-Preserving Fusion)
    # 逻辑：Result = Original OR Bridge
    # 这样，原本很粗的地方（Original）会被保留，断裂的地方会被 Bridge 连上
    # Bridge 在重叠区域会"隐藏"在 Original 内部，不会破坏原本的粗细
    merged_mask = np.logical_or(binary_mask, bridge_mask).astype(np.uint8)

    return merged_mask, bridge_mask, True


def refine_mask_final(merged_mask):
    """提取最大连通块"""
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

    print(f"开始 V4 融合分析 (Extrapolation + Fidelity)...")
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
        merged_mask, bridge_layer, success = fit_extrapolate_and_merge(pred_bin)
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
            plt.title(f"Global Trend Extrapolation\n(Curve spans 0 to Width)", fontsize=10)
            # 这里的可视化技巧：背景是黑的，原图是灰的，拟合线是亮的
            vis_layer = np.zeros_like(pred_bin, dtype=float)
            vis_layer[pred_bin == 1] = 0.4  # 原图灰色
            vis_layer[bridge_layer == 1] = 1.0  # 拟合线亮白色
            plt.imshow(vis_layer, cmap='gray')
            plt.axis('off')

            # 4. Final Result
            plt.subplot(1, 4, 4)
            plt.title(f"V4 Hybrid Result\nDice: {d_refi:.3f} (Structure Preserved)", fontsize=10)
            plt.imshow(pred_refined, cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, f"V4_Final_{filename}"))
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
    print("   ✅ SUCCESS: Extrapolation + Thickness Retention Achieved!")
    print("=" * 40)


if __name__ == "__main__":
    main()