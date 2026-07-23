import os
import cv2
import numpy as np
from skimage import measure
from glob import glob
from tqdm import tqdm

# ================= 配置区域 =================
# 这里直接指向你刚才生成的文件夹
PRED_DIR = r"/Project_Data/Predictions"

# ===========================================

def evaluate_fragmentation():
    files = sorted(glob(os.path.join(PRED_DIR, "*.png")))
    if not files:
        print(f"Error: 在 {PRED_DIR} 没找到图片！")
        return

    # 假设你的 Label 是 0-10 (共11类)
    # 如果你的类别更多，可以改大这个数字
    MAX_CLASS_ID = 11

    # 记录每一层的平均连通分量数量 (越大约碎)
    # 字典结构: {class_id: [comp_count_img1, comp_count_img2, ...]}
    class_fragmentation = {k: [] for k in range(1, MAX_CLASS_ID)}

    print(f"正在扫描 {len(files)} 张预测图，寻找最适合做实验的'破碎层'...")

    # 遍历所有图片
    for p_path in tqdm(files):
        # 读取预测图 (灰度模式)
        pred = cv2.imread(p_path, 0)

        if pred is None:
            continue

        # 遍历每一层 (1 到 10)
        for class_id in range(1, MAX_CLASS_ID):
            # 提取该层的二值图
            # 只有像素值等于当前 class_id 的地方是 1，其他是 0
            binary_layer = (pred == class_id).astype(np.uint8)

            # 如果这张图里根本没有这一层（比如切片切到了眼球边缘），跳过统计
            if np.sum(binary_layer) == 0:
                continue

            # 【核心数学计算】计算连通分量数量 b0
            # connectivity=2 对应 8-连通 (8-connectivity)
            _, num_features = measure.label(binary_layer, return_num=True, connectivity=2)

            class_fragmentation[class_id].append(num_features)

    print("\n" + "=" * 20 + " 拓扑审计报告 (Topological Audit Report) " + "=" * 20)
    best_candidate = -1
    max_frags = -1

    print(f"{'Class ID':<10} | {'Avg Components (b0)':<25} | {'Status'}")
    print("-" * 60)

    for class_id in range(1, MAX_CLASS_ID):
        frags = class_fragmentation[class_id]

        # 如果这一层在所有图里都没出现过
        if not frags:
            print(f"{class_id:<10} | {'N/A':<25} | Empty Layer")
            continue

        avg_frag = np.mean(frags)

        # 标记状态
        status = ""
        if avg_frag == 1.0:
            status = "Perfect (Too boring for project)"
        elif avg_frag > 1.5:
            status = ">> FRAGMENTED (Good Choice) <<"
        elif avg_frag > 3.0:
            status = ">> HIGHLY BROKEN (Best Choice) <<"

        print(f"{class_id:<10} | {avg_frag:.4f}{' ' * 20} | {status}")

        # 寻找破碎度最高的层
        if avg_frag > max_frags:
            max_frags = avg_frag
            best_candidate = class_id

    print("-" * 60)
    print(f"✅ 推荐使用的 Target Class ID: 【 {best_candidate} 】")
    print(f"原因: 它的平均连通分量数是 {max_frags:.2f} (理想值应为 1.0)。")
    print("这意味着这一层经常断裂成多段，非常适合用图论算法来修复")
    print("请记住这个 ID，填入下一步的核心代码中。")


if __name__ == "__main__":
    evaluate_fragmentation()