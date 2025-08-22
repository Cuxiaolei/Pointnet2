import os
import numpy as np
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
import provider


# 读取txt文件中的场景路径
def load_scene_list(txt_path):
    with open(txt_path, 'r') as f:
        scenes = [line.strip() for line in f if line.strip()]
    return scenes


class CustomScannetDataset():
    def __init__(self, root, split='train', npoints=8192, use_color=True, use_normal=True):
        self.root = root  # 数据集根目录（txt文件所在目录）
        self.split = split  # train/val/test
        self.npoints = npoints  # 每个样本的点数量
        self.use_color = use_color  # 是否使用颜色特征
        self.use_normal = use_normal  # 是否使用法向量特征

        # 加载场景列表（从txt文件）
        txt_file = os.path.join(root, f'{split}_scenes.txt')
        self.scene_paths = load_scene_list(txt_file)
        print(f"Loaded {len(self.scene_paths)} {split} scenes from {txt_file}")

        # 加载所有场景数据
        self.all_points = []  # 点云特征 (N, 3+3+3) 坐标+颜色+法向量
        self.all_labels = []  # 语义标签 (N,)

        for scene_path in self.scene_paths:
            # 加载.npy文件（格式：坐标3+颜色3+法向量3+标签1）
            full_path = os.path.join(root, scene_path)  # 拼接完整路径
            data = np.load(full_path)

            # 提取特征和标签
            points_xyz = data[:, 0:3]  # 坐标 (N,3)
            points_color = data[:, 3:6]  # 颜色 (N,3)
            points_normal = data[:, 6:9]  # 法向量 (N,3)
            labels = data[:, 9].astype(np.int32)  # 标签 (N,)

            # 组合特征（根据需要选择是否包含颜色和法向量）
            features = [points_xyz]
            if use_color:
                features.append(points_color)
            if use_normal:
                features.append(points_normal)
            full_features = np.concatenate(features, axis=1)  # (N, 3/6/9)

            self.all_points.append(full_features)
            self.all_labels.append(labels)

        # 计算标签权重（解决类别不平衡）
        self.num_classes = 3  # 三分类（电塔、背景、电力线）
        if split == 'train':
            label_counts = np.zeros(self.num_classes, dtype=np.float32)
            for labels in self.all_labels:
                counts = np.bincount(labels, minlength=self.num_classes)
                label_counts += counts
            label_weights = label_counts / np.sum(label_counts)
            self.labelweights = 1 / np.log(1.2 + label_weights)  # 权重计算
        else:
            self.labelweights = np.ones(self.num_classes)

    def __getitem__(self, index):
        # 获取单一场景数据
        point_set = self.all_points[index].copy()  # (N, 3/6/9)
        semantic_seg = self.all_labels[index].copy()  # (N,)

        # 计算坐标范围（用于裁剪子体积）
        xyz = point_set[:, 0:3]  # 坐标始终是前3维
        coordmax = np.max(xyz, axis=0)
        coordmin = np.min(xyz, axis=0)
        smp_size = np.array([1.5, 1.5, 3.0])  # 子体积大小（x,y,z）
        smpmin = np.maximum(coordmax - smp_size, coordmin)
        smpmin[2] = coordmin[2]  # z轴从最低点开始

        # 随机裁剪子体积（确保包含足够有效点）
        for _ in range(10):
            # 随机选择中心点
            curcenter = xyz[np.random.choice(len(xyz)), :]
            curmin = curcenter - [0.75, 0.75, 1.5]
            curmax = curcenter + [0.75, 0.75, 1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]

            # 筛选子体积内的点
            in_box = np.logical_and(xyz >= curmin - 0.2, xyz <= curmax + 0.2).all(axis=1)
            cur_points = point_set[in_box]
            cur_labels = semantic_seg[in_box]

            if len(cur_labels) == 0:
                continue  # 空体积则重试

            # 验证子体积有效性（有效点占比≥70%）
            valid_ratio = np.sum(cur_labels >= 0) / len(cur_labels)
            if valid_ratio >= 0.7:
                break

        # 采样固定数量的点
        if len(cur_points) >= self.npoints:
            choice = np.random.choice(len(cur_points), self.npoints, replace=False)
        else:
            choice = np.random.choice(len(cur_points), self.npoints, replace=True)
        cur_points = cur_points[choice]
        cur_labels = cur_labels[choice]

        # 数据增强（仅对坐标和法向量进行几何变换）
        if self.split == 'train':
            # 绕Z轴旋转
            cur_points[:, 0:3] = provider.rotate_point_cloud_z(cur_points[:, 0:3])

            # 若使用法向量，同步旋转法向量
            if self.use_normal and cur_points.shape[1] >= 6:
                # 提取旋转矩阵（来自provider的内部实现）
                theta = np.random.uniform(0, np.pi * 2)
                rotation_matrix = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
                # 旋转法向量（假设法向量在6-8列）
                cur_points[:, 6:9] = cur_points[:, 6:9] @ rotation_matrix.T

            # 随机缩放
            cur_points[:, 0:3] = provider.random_scale_point_cloud(cur_points[:, 0:3])

            # 随机平移
            cur_points[:, 0:3] = provider.shift_point_cloud(cur_points[:, 0:3])

            # 坐标抖动（避免过度影响纤细结构如电力线）
            cur_points[:, 0:3] = provider.jitter_point_cloud(
                cur_points[:, 0:3], sigma=0.01, clip=0.05
            )

        # 计算样本权重
        sample_weight = self.labelweights[cur_labels]

        return cur_points, cur_labels, sample_weight

    def __len__(self):
        return len(self.scene_paths)


class CustomScannetDatasetWholeScene():
    """全场景测试数据集（不裁剪子体积，用于最终评估）"""

    def __init__(self, root, split='test', npoints=8192, use_color=True, use_normal=True):
        self.root = root
        self.split = split
        self.npoints = npoints
        self.use_color = use_color
        self.use_normal = use_normal

        # 加载场景列表
        txt_file = os.path.join(root, f'{split}_scenes.txt')
        self.scene_paths = load_scene_list(txt_file)
        print(f"Loaded {len(self.scene_paths)} {split} scenes from {txt_file}")

        # 加载数据
        self.all_points = []
        self.all_labels = []
        for scene_path in self.scene_paths:
            full_path = os.path.join(root, scene_path)
            data = np.load(full_path)

            points_xyz = data[:, 0:3]
            points_color = data[:, 3:6]
            points_normal = data[:, 6:9]
            labels = data[:, 9].astype(np.int32)

            features = [points_xyz]
            if use_color:
                features.append(points_color)
            if use_normal:
                features.append(points_normal)
            full_features = np.concatenate(features, axis=1)

            self.all_points.append(full_features)
            self.all_labels.append(labels)

        self.num_classes = 3
        self.labelweights = np.ones(self.num_classes)

    def __getitem__(self, index):
        point_set = self.all_points[index].copy()
        semantic_seg = self.all_labels[index].copy()
        xyz = point_set[:, 0:3]

        # 计算场景分割的子体积网格
        coordmax = np.max(xyz, axis=0)
        coordmin = np.min(xyz, axis=0)
        nsubvolume_x = int(np.ceil((coordmax[0] - coordmin[0]) / 1.5))
        nsubvolume_y = int(np.ceil((coordmax[1] - coordmin[1]) / 1.5))

        sub_clouds = []
        sub_labels = []
        sub_weights = []

        # 遍历所有子体积
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin + [i * 1.5, j * 1.5, 0]
                curmax = coordmin + [(i + 1) * 1.5, (j + 1) * 1.5, coordmax[2] - coordmin[2]]

                # 筛选子体积内的点
                in_box = np.logical_and(xyz >= curmin - 0.2, xyz <= curmax + 0.2).all(axis=1)
                cur_points = point_set[in_box]
                cur_labels = semantic_seg[in_box]

                if len(cur_labels) == 0:
                    continue

                # 采样点
                if len(cur_points) >= self.npoints:
                    choice = np.random.choice(len(cur_points), self.npoints, replace=False)
                else:
                    choice = np.random.choice(len(cur_points), self.npoints, replace=True)
                cur_points = cur_points[choice]
                cur_labels = cur_labels[choice]

                # 计算权重
                sample_weight = self.labelweights[cur_labels]

                sub_clouds.append(np.expand_dims(cur_points, 0))
                sub_labels.append(np.expand_dims(cur_labels, 0))
                sub_weights.append(np.expand_dims(sample_weight, 0))

        # 拼接所有子体积
        sub_clouds = np.concatenate(sub_clouds, axis=0) if sub_clouds else np.array([])
        sub_labels = np.concatenate(sub_labels, axis=0) if sub_labels else np.array([])
        sub_weights = np.concatenate(sub_weights, axis=0) if sub_weights else np.array([])

        return sub_clouds, sub_labels, sub_weights

    def __len__(self):
        return len(self.scene_paths)
