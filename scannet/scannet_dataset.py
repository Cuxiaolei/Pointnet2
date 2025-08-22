import os
import numpy as np
import sys
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
import provider


def load_scene_list(txt_path):
    with open(txt_path, 'r') as f:
        scenes = [line.strip() for line in f if line.strip()]
    return scenes


class CustomScannetDataset():
    def __init__(self, root, split='train', npoints=8192, use_color=True, use_normal=True):
        self.root = root
        self.split = split
        self.npoints = npoints
        self.use_color = use_color
        self.use_normal = use_normal

        txt_file = os.path.join(root, f'{split}_scenes.txt')
        self.scene_paths = load_scene_list(txt_file)
        print(f"Loaded {len(self.scene_paths)} {split} scenes from {txt_file}")

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
        if split == 'train':
            label_counts = np.zeros(self.num_classes, dtype=np.float32)
            for labels in self.all_labels:
                counts = np.bincount(labels, minlength=self.num_classes)
                label_counts += counts
            label_weights = label_counts / np.sum(label_counts)
            self.labelweights = 1 / np.log(1.2 + label_weights)
        else:
            self.labelweights = np.ones(self.num_classes)

    def __getitem__(self, index):
        point_set = self.all_points[index].copy()
        semantic_seg = self.all_labels[index].copy()

        xyz = point_set[:, 0:3]
        coordmax = np.max(xyz, axis=0)
        coordmin = np.min(xyz, axis=0)
        smp_size = np.array([1.5, 1.5, 3.0])
        smpmin = np.maximum(coordmax - smp_size, coordmin)
        smpmin[2] = coordmin[2]

        for _ in range(10):
            curcenter = xyz[np.random.choice(len(xyz)), :]
            curmin = curcenter - [0.75, 0.75, 1.5]
            curmax = curcenter + [0.75, 0.75, 1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]

            in_box = np.logical_and(xyz >= curmin - 0.2, xyz <= curmax + 0.2).all(axis=1)
            cur_points = point_set[in_box]
            cur_labels = semantic_seg[in_box]

            if len(cur_labels) == 0:
                continue

            valid_ratio = np.sum(cur_labels >= 0) / len(cur_labels)
            if valid_ratio >= 0.7:
                break

        if len(cur_points) >= self.npoints:
            choice = np.random.choice(len(cur_points), self.npoints, replace=False)
        else:
            choice = np.random.choice(len(cur_points), self.npoints, replace=True)
        cur_points = cur_points[choice]
        cur_labels = cur_labels[choice]

        if self.split == 'train':
            cur_points[:, 0:3] = provider.rotate_point_cloud_z(cur_points[:, 0:3])

            if self.use_normal and cur_points.shape[1] >= 6:
                theta = np.random.uniform(0, np.pi * 2)
                rotation_matrix = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
                cur_points[:, 6:9] = cur_points[:, 6:9] @ rotation_matrix.T

            cur_points[:, 0:3] = provider.random_scale_point_cloud(cur_points[:, 0:3])
            cur_points[:, 0:3] = provider.shift_point_cloud(cur_points[:, 0:3])
            cur_points[:, 0:3] = provider.jitter_point_cloud(
                cur_points[:, 0:3], sigma=0.01, clip=0.05
            )

        sample_weight = self.labelweights[cur_labels]

        return cur_points, cur_labels, sample_weight

    def __len__(self):
        return len(self.scene_paths)


class CustomScannetDatasetWholeScene():
    def __init__(self, root, split='test', npoints=8192, use_color=True, use_normal=True):
        self.root = root
        self.split = split
        self.npoints = npoints
        self.use_color = use_color
        self.use_normal = use_normal

        txt_file = os.path.join(root, f'{split}_scenes.txt')
        self.scene_paths = load_scene_list(txt_file)
        print(f"Loaded {len(self.scene_paths)} {split} scenes from {txt_file}")

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

        coordmax = np.max(xyz, axis=0)
        coordmin = np.min(xyz, axis=0)
        nsubvolume_x = int(np.ceil((coordmax[0] - coordmin[0]) / 1.5))
        nsubvolume_y = int(np.ceil((coordmax[1] - coordmin[1]) / 1.5))

        sub_clouds = []
        sub_labels = []
        sub_weights = []

        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin + [i * 1.5, j * 1.5, 0]
                curmax = coordmin + [(i + 1) * 1.5, (j + 1) * 1.5, coordmax[2] - coordmin[2]]

                in_box = np.logical_and(xyz >= curmin - 0.2, xyz <= curmax + 0.2).all(axis=1)
                cur_points = point_set[in_box]
                cur_labels = semantic_seg[in_box]

                if len(cur_labels) == 0:
                    continue

                if len(cur_points) >= self.npoints:
                    choice = np.random.choice(len(cur_points), self.npoints, replace=False)
                else:
                    choice = np.random.choice(len(cur_points), self.npoints, replace=True)
                cur_points = cur_points[choice]
                cur_labels = cur_labels[choice]

                sample_weight = self.labelweights[cur_labels]

                sub_clouds.append(np.expand_dims(cur_points, 0))
                sub_labels.append(np.expand_dims(cur_labels, 0))
                sub_weights.append(np.expand_dims(sample_weight, 0))

        sub_clouds = np.concatenate(sub_clouds, axis=0) if sub_clouds else np.array([])
        sub_labels = np.concatenate(sub_labels, axis=0) if sub_labels else np.array([])
        sub_weights = np.concatenate(sub_weights, axis=0) if sub_weights else np.array([])

        return sub_clouds, sub_labels, sub_weights

    def __len__(self):
        return len(self.scene_paths)
