import argparse
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import pc_util

sys.path.append(os.path.join(ROOT_DIR, 'data_prep'))
import scannet_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_sem_seg', help='Model name [default: pointnet2_sem_seg]')
parser.add_argument('--log_dir', default='log_power_line', help='Log dir [default: log_power_line]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=150, help='Epoch to run [default: 150]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Initial learning rate [default: 0.0005]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=100000, help='Decay step for lr decay [default: 100000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model)
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))
os.system('cp train.py %s' % (LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

# 类别名称映射（与你的标签对应）
CLASS_NAMES = ['background', 'tower', 'power_line']  # 0:背景, 1:电塔, 2:电力线
NUM_CLASSES = len(CLASS_NAMES)

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

DATA_PATH = os.path.join(ROOT_DIR, 'data', 'your_power_dataset')

TRAIN_DATASET = scannet_dataset.CustomScannetDataset(
    root=DATA_PATH,
    split='train',
    npoints=NUM_POINT,
    use_color=True,
    use_normal=True
)
TEST_DATASET = scannet_dataset.CustomScannetDataset(
    root=DATA_PATH,
    split='test',
    npoints=NUM_POINT,
    use_color=True,
    use_normal=True
)
TEST_DATASET_WHOLE_SCENE = scannet_dataset.CustomScannetDatasetWholeScene(
    root=DATA_PATH,
    split='test',
    npoints=NUM_POINT,
    use_color=True,
    use_normal=True
)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# 计算IoU的工具函数
def compute_iou(pred, label, num_classes):
    """
    计算每个类别的IoU和平均mIoU
    pred: (B*N,) 预测标签
    label: (B*N,) 真实标签
    return: 每个类别的IoU列表, mIoU
    """
    ious = []
    for cls in range(num_classes):
        # 真实为cls且预测为cls (TP)
        tp = np.sum((pred == cls) & (label == cls))
        # 预测为cls但真实不是cls (FP)
        fp = np.sum((pred == cls) & (label != cls))
        # 真实为cls但预测不是cls (FN)
        fn = np.sum((pred != cls) & (label == cls))
        # IoU = TP / (TP + FP + FN)，避免除以零
        if tp + fp + fn == 0:
            iou = 0.0
        else:
            iou = tp / (tp + fp + fn)
        ious.append(iou)
    mIoU = np.mean(ious)
    return ious, mIoU


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,
        batch * BATCH_SIZE,
        DECAY_STEP,
        DECAY_RATE,
        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001)
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, smpws_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE * NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'smpws_pl': smpws_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_mIoU = -1
        for epoch in range(MAX_EPOCH):
            log_string('=' * 50)
            log_string(f'**** EPOCH {epoch:03d} / {MAX_EPOCH:03d} ****')
            log_string(f'当前时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            sys.stdout.flush()

            # 训练一轮并获取训练集指标
            train_loss, train_acc, train_ious, train_mIoU = train_one_epoch(sess, ops, train_writer)
            log_string('=' * 30)
            log_string(f'训练集指标 - EPOCH {epoch:03d}')
            log_string(f'平均损失: {train_loss:.4f}')
            log_string(f'整体准确率: {train_acc:.4f}')
            log_string(f'每个类别的IoU:')
            for cls_idx in range(NUM_CLASSES):
                log_string(f'  {CLASS_NAMES[cls_idx]}: {train_ious[cls_idx]:.4f}')
            log_string(f'平均mIoU: {train_mIoU:.4f}')

            # 每5轮评估测试集
            if epoch % 5 == 0:
                log_string('-' * 30)
                log_string(f'测试集指标 (随机裁剪) - EPOCH {epoch:03d}')
                test_acc, test_ious, test_mIoU = eval_one_epoch(sess, ops, test_writer)

                log_string('-' * 30)
                log_string(f'测试集指标 (全场景) - EPOCH {epoch:03d}')
                whole_acc, whole_ious, whole_mIoU = eval_whole_scene_one_epoch(sess, ops, test_writer)

                # 保存最优模型（基于全场景mIoU）
                if whole_mIoU > best_mIoU:
                    best_mIoU = whole_mIoU
                    save_path = saver.save(sess, os.path.join(LOG_DIR, f"best_model_epoch_{epoch:03d}.ckpt"))
                    log_string(f'最佳模型已保存至: {save_path} (最佳mIoU: {best_mIoU:.4f})')

            # 每10轮保存一次模型
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string(f'模型已保存至: {save_path}')
            log_string('=' * 50 + '\n')


def get_batch_wdp(dataset, idxs, start_idx, end_idx):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 9))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps, seg, smpw = dataset[idxs[i + start_idx]]
        batch_data[i, ...] = ps
        batch_label[i, :] = seg
        batch_smpw[i, :] = smpw

        dropout_ratio = np.random.random() * 0.875
        drop_idx = np.where(np.random.random((ps.shape[0])) <= dropout_ratio)[0]
        batch_data[i, drop_idx, 0:3] = batch_data[i, 0, 0:3]
        batch_label[i, drop_idx] = batch_label[i, 0]
        batch_smpw[i, drop_idx] *= 0
    return batch_data, batch_label, batch_smpw


def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 9))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps, seg, smpw = dataset[idxs[i + start_idx]]
        batch_data[i, ...] = ps
        batch_label[i, :] = seg
        batch_smpw[i, :] = smpw
    return batch_data, batch_label, batch_smpw


def train_one_epoch(sess, ops, train_writer):
    is_training = True
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET) // BATCH_SIZE

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    all_pred = []
    all_label = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        batch_data, batch_label, batch_smpw = get_batch_wdp(TRAIN_DATASET, train_idxs, start_idx, end_idx)

        aug_data = batch_data.copy()
        aug_data[:, :, 0:3] = provider.rotate_point_cloud_z(aug_data[:, :, 0:3])

        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training, }
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred']],
                                                        feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        # 收集所有预测和标签用于计算IoU
        pred_val = np.argmax(pred_val, 2)  # (B, N)
        all_pred.append(pred_val.reshape(-1))  # 展平为(B*N,)
        all_label.append(batch_label.reshape(-1))

        # 统计损失和准确率
        total_loss += loss_val
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE * NUM_POINT)

        # 每10个批次打印一次中间结果
        if (batch_idx + 1) % 10 == 0:
            log_string(
                f'  批次 {batch_idx + 1:03d}/{num_batches:03d} - 损失: {loss_val:.4f}, 准确率: {correct / (BATCH_SIZE * NUM_POINT):.4f}')

    # 计算整个训练集的指标
    avg_loss = total_loss / num_batches
    avg_acc = total_correct / total_seen
    all_pred = np.concatenate(all_pred, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    ious, mIoU = compute_iou(all_pred, all_label, NUM_CLASSES)

    return avg_loss, avg_acc, ious, mIoU


def eval_one_epoch(sess, ops, test_writer):
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = len(TEST_DATASET) // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    all_pred = []
    all_label = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        batch_data, batch_label, batch_smpw = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)

        aug_data = batch_data.copy()
        aug_data[:, :, 0:3] = provider.rotate_point_cloud_z(aug_data[:, :, 0:3])

        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)

        pred_val = np.argmax(pred_val, 2)
        all_pred.append(pred_val.reshape(-1))
        all_label.append(batch_label.reshape(-1))

        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE * NUM_POINT)

    # 计算测试集指标
    avg_acc = total_correct / total_seen
    all_pred = np.concatenate(all_pred, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    ious, mIoU = compute_iou(all_pred, all_label, NUM_CLASSES)

    # 输出详细指标
    log_string(f'平均准确率: {avg_acc:.4f}')
    log_string(f'每个类别的IoU:')
    for cls_idx in range(NUM_CLASSES):
        log_string(f'  {CLASS_NAMES[cls_idx]}: {ious[cls_idx]:.4f}')
    log_string(f'平均mIoU: {mIoU:.4f}')

    EPOCH_CNT += 1
    return avg_acc, ious, mIoU


def eval_whole_scene_one_epoch(sess, ops, test_writer):
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET_WHOLE_SCENE))
    num_batches = len(TEST_DATASET_WHOLE_SCENE)

    total_correct = 0
    total_seen = 0
    all_pred = []
    all_label = []

    is_continue_batch = False
    extra_batch_data = np.zeros((0, NUM_POINT, 9))
    extra_batch_label = np.zeros((0, NUM_POINT))
    extra_batch_smpw = np.zeros((0, NUM_POINT))

    for batch_idx in range(num_batches):
        if not is_continue_batch:
            batch_data, batch_label, batch_smpw = TEST_DATASET_WHOLE_SCENE[batch_idx]
            batch_data = np.concatenate((batch_data, extra_batch_data), axis=0)
            batch_label = np.concatenate((batch_label, extra_batch_label), axis=0)
            batch_smpw = np.concatenate((batch_smpw, extra_batch_smpw), axis=0)
        else:
            batch_data_tmp, batch_label_tmp, batch_smpw_tmp = TEST_DATASET_WHOLE_SCENE[batch_idx]
            batch_data = np.concatenate((batch_data, batch_data_tmp), axis=0)
            batch_label = np.concatenate((batch_label, batch_label_tmp), axis=0)
            batch_smpw = np.concatenate((batch_smpw, batch_smpw_tmp), axis=0)

        if batch_data.shape[0] < BATCH_SIZE:
            is_continue_batch = True
            continue
        elif batch_data.shape[0] == BATCH_SIZE:
            is_continue_batch = False
            extra_batch_data = np.zeros((0, NUM_POINT, 9))
            extra_batch_label = np.zeros((0, NUM_POINT))
            extra_batch_smpw = np.zeros((0, NUM_POINT))
        else:
            is_continue_batch = False
            extra_batch_data = batch_data[BATCH_SIZE:, :, :]
            extra_batch_label = batch_label[BATCH_SIZE:, :]
            extra_batch_smpw = batch_smpw[BATCH_SIZE:, :]
            batch_data = batch_data[:BATCH_SIZE, :, :]
            batch_label = batch_label[:BATCH_SIZE, :]
            batch_smpw = batch_smpw[:BATCH_SIZE, :]

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)

        pred_val = np.argmax(pred_val, 2)
        all_pred.append(pred_val.reshape(-1))
        all_label.append(batch_label.reshape(-1))

        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (batch_data.shape[0] * NUM_POINT)

    # 计算全场景指标
    avg_acc = total_correct / total_seen
    all_pred = np.concatenate(all_pred, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    ious, mIoU = compute_iou(all_pred, all_label, NUM_CLASSES)

    # 输出详细指标
    log_string(f'平均准确率: {avg_acc:.4f}')
    log_string(f'每个类别的IoU:')
    for cls_idx in range(NUM_CLASSES):
        log_string(f'  {CLASS_NAMES[cls_idx]}: {ious[cls_idx]:.4f}')
    log_string(f'平均mIoU: {mIoU:.4f}')

    EPOCH_CNT += 1
    return avg_acc, ious, mIoU


if __name__ == "__main__":
    log_string('训练开始，进程ID: %s' % (str(os.getpid())))
    log_string(f'类别映射: {CLASS_NAMES} (共{NUM_CLASSES}类)')
    train()
    LOG_FOUT.close()