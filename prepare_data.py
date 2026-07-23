import os
import shutil
from glob import glob
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 你的验证集输出文件夹
VALIDATION_RAW_DIR = r"D:\nnUNet_results\Dataset001_OCT\nnUNetTrainer__nnUNetPlans__2d\fold_2\validation"

# 2. 你的所有原始标签文件夹
ALL_GT_DIR = r"D:\nnUNet_raw\Dataset001_OCT\labelsTr"

# 3. 准备输出的新文件夹
OUTPUT_PRED_DIR = r"./Project_Data/Predictions"
OUTPUT_GT_DIR = r"./Project_Data/GroundTruth"


# ===========================================

def prepare_data():
    # 创建输出目录
    os.makedirs(OUTPUT_PRED_DIR, exist_ok=True)
    os.makedirs(OUTPUT_GT_DIR, exist_ok=True)

    # 1. 扫描验证集中的 PNG 文件 (排除 npz 和 pkl)
    # 假设文件名格式是 case_xxxxx.png
    pred_files = sorted(glob(os.path.join(VALIDATION_RAW_DIR, "*.png")))

    if len(pred_files) == 0:
        print("错误：在验证集文件夹里没找到 .png 文件！请检查路径。")
        return

    print(f"找到 {len(pred_files)} 张预测图片，准备开始清洗和配对...")

    matched_count = 0
    missing_count = 0

    for pred_path in tqdm(pred_files):
        filename = os.path.basename(pred_path)  # 获取文件名，如 case_0011.png

        # --- 步骤 A: 复制预测图到新文件夹 ---
        shutil.copy(pred_path, os.path.join(OUTPUT_PRED_DIR, filename))

        # --- 步骤 B: 去总库里找对应的 GT ---
        gt_source_path = os.path.join(ALL_GT_DIR, filename)

        if os.path.exists(gt_source_path):
            shutil.copy(gt_source_path, os.path.join(OUTPUT_GT_DIR, filename))
            matched_count += 1
        else:
            print(f"警告：找不到对应的 GT 文件 -> {filename}")
            missing_count += 1

    print("\n" + "=" * 30)
    print("数据准备完成！")
    print(f"成功配对: {matched_count} 对")
    if missing_count > 0:
        print(f"缺失 GT: {missing_count} 个 (请检查 labelsTr 是否完整)")
    print(f"预测集已存入: {OUTPUT_PRED_DIR}")
    print(f"真值集已存入: {OUTPUT_GT_DIR}")
    print("=" * 30)
    print("现在，请把上面两个路径填入 topo_analysis.py 开始跑核心实验")


if __name__ == "__main__":
    prepare_data()