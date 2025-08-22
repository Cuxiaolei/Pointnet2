import os
import numpy as np

# -------------------------- 配置区域（在此处修改你的文件路径和预期标签） --------------------------
# 可以是单个.npy文件路径（如："data/scene0001.npy"）或存放.npy文件的文件夹路径（如："data/scenes/"）
TARGET_PATH = r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\output\merged"  # 替换为你的文件/文件夹路径
EXPECTED_CLASSES = [0, 1, 2]  # 替换为你的预期标签列表（如[10, 20, 30]）

# ------------------------------------------------------------------------------------------------

def check_scene_data(scene_path, expected_classes):
    """检查单个场景的.npy文件是否符合要求（坐标3+颜色3+法向量3+标签1，共10维）"""
    try:
        data = np.load(scene_path)
    except Exception as e:
        return False, f"加载失败: {str(e)}"

    # 检查数据维度
    if data.ndim != 2:
        return False, f"数据应为2维数组 (N点 x 10特征)，实际为 {data.ndim} 维"
    if data.shape[1] != 10:
        return False, f"每个点需包含10维特征(3坐标+3颜色+3法向量+1标签)，实际维度: {data.shape[1]}"

    # 提取字段
    xyz = data[:, 0:3]  # 坐标 (0-2列)
    colors = data[:, 3:6]  # 颜色 (3-5列)
    normals = data[:, 6:9]  # 法向量 (6-8列)
    semantic_labels = data[:, 9].astype(np.int32)  # 标签 (第9列)

    # 检查异常值
    for name, arr in [("坐标", xyz), ("颜色", colors), ("法向量", normals)]:
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            return False, f"{name}包含NaN或无穷大值"

    # 检查颜色范围（通常为[0,1]或[0,255]）
    color_min, color_max = colors.min(), colors.max()
    if not ((0 <= color_min and color_max <= 1) or (0 <= color_min and color_max <= 255)):
        return False, f"颜色值范围异常 [{color_min:.2f}, {color_max:.2f}]，建议[0,1]或[0,255]"

    # 检查法向量范围（通常为[-1,1]）
    normal_min, normal_max = normals.min(), normals.max()
    if not (-1.1 <= normal_min and normal_max <= 1.1):  # 允许微小浮点误差
        return False, f"法向量范围异常 [{normal_min:.2f}, {normal_max:.2f}]，建议[-1,1]"

    # 检查标签有效性
    unique_labels = np.unique(semantic_labels)
    invalid_labels = [l for l in unique_labels if l not in expected_classes]
    if len(invalid_labels) > 0:
        return False, f"存在无效标签 {invalid_labels}，预期标签为 {expected_classes}"

    # 检查点数量
    if len(data) < 100:
        return False, f"点数量过少 ({len(data)}个)，可能存在数据缺失"

    return True, (
        f"检查通过 | 点数量: {len(data)} | "
        f"坐标范围: X[{xyz[:, 0].min():.2f},{xyz[:, 0].max():.2f}], "
        f"Y[{xyz[:, 1].min():.2f},{xyz[:, 1].max():.2f}], "
        f"Z[{xyz[:, 2].min():.2f},{xyz[:, 2].max():.2f}] | "
        f"标签分布: {dict(zip(unique_labels, np.bincount(semantic_labels)))}"
    )


def main():
    # 收集需要检查的文件列表
    if os.path.isfile(TARGET_PATH) and TARGET_PATH.endswith(".npy"):
        # 输入为单个.npy文件
        scene_files = [TARGET_PATH]
    elif os.path.isdir(TARGET_PATH):
        # 输入为文件夹，遍历所有.npy文件
        scene_files = [
            os.path.join(TARGET_PATH, f)
            for f in os.listdir(TARGET_PATH)
            if f.endswith(".npy") and os.path.isfile(os.path.join(TARGET_PATH, f))
        ]
    else:
        print(f"错误：路径 {TARGET_PATH} 不是有效的.npy文件或文件夹")
        return

    if not scene_files:
        print(f"错误：在 {TARGET_PATH} 中未找到.npy文件")
        return

    # 批量检查
    total = len(scene_files)
    valid = 0
    print(f"开始检查 {total} 个场景文件...\n")
    for i, file_path in enumerate(scene_files, 1):
        filename = os.path.basename(file_path)
        success, msg = check_scene_data(file_path, EXPECTED_CLASSES)
        status = "✅" if success else "❌"
        print(f"[{i}/{total}] {status} {filename}: {msg}")
        if success:
            valid += 1

    # 输出总结
    print(f"\n检查完成 | 有效文件: {valid}/{total} | 有效率: {valid / total:.2%}")
    if valid < total:
        print("警告：存在无效文件，请检查上述错误信息并修正")


if __name__ == "__main__":
    main()