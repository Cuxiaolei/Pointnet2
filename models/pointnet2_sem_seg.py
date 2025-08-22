import os
import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module


def placeholder_inputs(batch_size, num_point):
    # 修改：输入维度从3改为9（3坐标+3颜色+3法向量）
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl


def get_model(point_cloud, is_training, num_class, bn_decay=None):
    """ Semantic segmentation PointNet++, input is BxNx9 (3坐标+3颜色+3法向量), output BxNxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    # 分离坐标和特征（颜色+法向量）
    l0_xyz = point_cloud[:, :, 0:3]  # 坐标始终是前3维
    l0_points = point_cloud[:, :, 3:9]  # 颜色（3-5）+法向量（6-8），共6维特征
    end_points['l0_xyz'] = l0_xyz

    # 第一层：使用9维特征（3坐标+6特征）进行采样和分组
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(
        l0_xyz, l0_points,
        npoint=1024, radius=0.1, nsample=32,
        mlp=[32, 32, 64],  # 输入特征维度为6（颜色+法向量），通过MLP处理
        mlp2=None, group_all=False,
        is_training=is_training, bn_decay=bn_decay, scope='layer1'
    )

    # 第二层
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(
        l1_xyz, l1_points,
        npoint=256, radius=0.2, nsample=32,
        mlp=[64, 64, 128],
        mlp2=None, group_all=False,
        is_training=is_training, bn_decay=bn_decay, scope='layer2'
    )

    # 第三层
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(
        l2_xyz, l2_points,
        npoint=64, radius=0.4, nsample=32,
        mlp=[128, 128, 256],
        mlp2=None, group_all=False,
        is_training=is_training, bn_decay=bn_decay, scope='layer3'
    )

    # 第四层
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(
        l3_xyz, l3_points,
        npoint=16, radius=0.8, nsample=32,
        mlp=[256, 256, 512],
        mlp2=None, group_all=False,
        is_training=is_training, bn_decay=bn_decay, scope='layer4'
    )

    # 特征传播层（上采样）
    l3_points = pointnet_fp_module(
        l3_xyz, l4_xyz, l3_points, l4_points,
        [256, 256], is_training, bn_decay, scope='fa_layer1'
    )

    l2_points = pointnet_fp_module(
        l2_xyz, l3_xyz, l2_points, l3_points,
        [256, 256], is_training, bn_decay, scope='fa_layer2'
    )

    l1_points = pointnet_fp_module(
        l1_xyz, l2_xyz, l1_points, l2_points,
        [256, 128], is_training, bn_decay, scope='fa_layer3'
    )

    l0_points = pointnet_fp_module(
        l0_xyz, l1_xyz, l0_points, l1_points,
        [128, 128, 128], is_training, bn_decay, scope='fa_layer4'
    )

    # 最终分类层
    net = tf_util.conv1d(
        l0_points, 128, 1,
        padding='VALID', bn=True,
        is_training=is_training, scope='fc1', bn_decay=bn_decay
    )
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(
        net, num_class, 1,
        padding='VALID', activation_fn=None, scope='fc2'
    )

    return net, end_points


def get_loss(pred, label, smpw):
    """ pred: BxNxC, label: BxN, smpw: BxN """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=label, logits=pred, weights=smpw
    )
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__ == '__main__':
    with tf.Graph().as_default():
        # 测试：输入维度改为9
        inputs = tf.zeros((32, 2048, 9))
        net, _ = get_model(inputs, tf.constant(True), 3)  # 3分类
        print(net)  # 应输出 (32, 2048, 3)
