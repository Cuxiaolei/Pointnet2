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
sys.path.append(os.path.join(ROOT_DIR, 'models'))  # 添加models目录
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

# 配置GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[FLAGS.gpu], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[FLAGS.gpu], True)
    except RuntimeError as e:
        print(e)

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
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
os.system(f'cp {MODEL_FILE} {LOG_DIR}')
os.system(f'cp train.py {LOG_DIR}')
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

# 类别名称映射
CLASS_NAMES = ['background', 'tower', 'power_line']
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


def compute_iou(pred, label, num_classes):
    ious = []
    for cls in range(num_classes):
        tp = np.sum((pred == cls) & (label == cls))
        fp = np.sum((pred == cls) & (label != cls))
        fn = np.sum((pred != cls) & (label == cls))
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        ious.append(iou)
    mIoU = np.mean(ious)
    return ious, mIoU


def get_learning_rate_scheduler():
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=BASE_LEARNING_RATE,
        decay_steps=DECAY_STEP,
        decay_rate=DECAY_RATE,
        staircase=True
    )

def get_bn_decay(batch):
    bn_momentum = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=BN_INIT_DECAY,
        decay_steps=BN_DECAY_DECAY_STEP,
        decay_rate=BN_DECAY_DECAY_RATE,
        staircase=True
    )(batch)
    return tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)


@tf.function
def train_step(pointclouds, labels, smpws, model, optimizer, bn_decay):
    with tf.GradientTape() as tape:
        pred, _ = model(pointclouds, training=True, bn_decay=bn_decay)
        loss = MODEL.get_loss(pred, labels, smpws)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    correct = tf.equal(tf.argmax(pred, 2), tf.cast(labels, tf.int64))
    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / (BATCH_SIZE * NUM_POINT)
    return loss, accuracy, pred


@tf.function
def test_step(pointclouds, labels, smpws, model, bn_decay):
    pred, _ = model(pointclouds, training=False, bn_decay=bn_decay)
    loss = MODEL.get_loss(pred, labels, smpws)
    correct = tf.equal(tf.argmax(pred, 2), tf.cast(labels, tf.int64))
    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / (BATCH_SIZE * NUM_POINT)
    return loss, accuracy, pred


def train():
    # 构建模型
    model = MODEL.get_model(NUM_CLASSES)

    # 优化器
    lr_scheduler = get_learning_rate_scheduler()
    if OPTIMIZER == 'momentum':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr_scheduler,
            momentum=MOMENTUM
        )
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_scheduler
        )

    # 日志记录
    train_summary_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, 'train'))
    test_summary_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, 'test'))

    best_mIoU = -1
    global_batch = 0

    for epoch in range(MAX_EPOCH):
        log_string('=' * 50)
        log_string(f'**** EPOCH {epoch:03d} / {MAX_EPOCH:03d} ****')
        log_string(f'当前时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        sys.stdout.flush()

        # 训练轮次
        train_losses = []
        train_accs = []
        train_preds = []
        train_labels = []

        train_idxs = np.arange(len(TRAIN_DATASET))
        np.random.shuffle(train_idxs)
        num_batches = len(TRAIN_DATASET) // BATCH_SIZE

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            batch_data, batch_label, batch_smpw = get_batch_wdp(TRAIN_DATASET, train_idxs, start_idx, end_idx)

            bn_decay = get_bn_decay(global_batch)
            loss, acc, pred = train_step(
                tf.convert_to_tensor(batch_data, dtype=tf.float32),
                tf.convert_to_tensor(batch_label, dtype=tf.int32),
                tf.convert_to_tensor(batch_smpw, dtype=tf.float32),
                model,
                optimizer,
                bn_decay
            )

            train_losses.append(loss.numpy())
            train_accs.append(acc.numpy())
            train_preds.append(pred.numpy().reshape(-1, NUM_CLASSES))
            train_labels.append(batch_label.reshape(-1))

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=global_batch)
                tf.summary.scalar('accuracy', acc, step=global_batch)
                tf.summary.scalar('bn_decay', bn_decay, step=global_batch)
                tf.summary.scalar('learning_rate', optimizer.lr(global_batch), step=global_batch)

            global_batch += 1

        # 计算训练集指标
        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_accs)
        all_pred = np.argmax(np.concatenate(train_preds, axis=0), axis=1)
        all_label = np.concatenate(train_labels, axis=0)
        train_ious, train_mIoU = compute_iou(all_pred, all_label, NUM_CLASSES)

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
            # 随机裁剪评估
            test_acc, test_ious, test_mIoU = eval_one_epoch(model, test_summary_writer, epoch, is_whole_scene=False)
            # 全场景评估
            whole_acc, whole_ious, whole_mIoU = eval_one_epoch(model, test_summary_writer, epoch, is_whole_scene=True)

            # 保存最佳模型
            if whole_mIoU > best_mIoU:
                best_mIoU = whole_mIoU
                save_path = os.path.join(LOG_DIR, f"best_model_epoch_{epoch:03d}")
                tf.saved_model.save(model, save_path)
                log_string(f'最佳模型已保存至: {save_path} (最佳mIoU: {best_mIoU:.4f})')

        # 每10轮保存一次模型
        if epoch % 10 == 0:
            save_path = os.path.join(LOG_DIR, f"model_epoch_{epoch:03d}")
            tf.saved_model.save(model, save_path)
            log_string(f'模型已保存至: {save_path}')

        log_string('=' * 50 + '\n')


def eval_one_epoch(model, summary_writer, epoch, is_whole_scene=False):
    dataset = TEST_DATASET_WHOLE_SCENE if is_whole_scene else TEST_DATASET
    preds = []
    labels = []
    accs = []

    for idx in range(len(dataset)):
        if is_whole_scene:
            batch_data, batch_label, batch_smpw = dataset[idx]
            batch_data = tf.convert_to_tensor(batch_data, dtype=tf.float32)
            batch_label = tf.convert_to_tensor(batch_label, dtype=tf.int32)
            batch_smpw = tf.convert_to_tensor(batch_smpw, dtype=tf.float32)

            pred, _ = model(batch_data, training=False, bn_decay=1.0)
            pred = tf.argmax(pred, axis=2).numpy()
            batch_label = batch_label.numpy()

            for b in range(pred.shape[0]):
                mask = batch_smpw[b] > 0
                if np.sum(mask) == 0:
                    continue
                correct = np.sum(pred[b, mask] == batch_label[b, mask])
                accs.append(correct / np.sum(mask))
                preds.append(pred[b, mask])
                labels.append(batch_label[b, mask])
        else:
            batch_data, batch_label, batch_smpw = dataset[idx]
            batch_data = tf.convert_to_tensor(batch_data[np.newaxis, ...], dtype=tf.float32)
            batch_label = tf.convert_to_tensor(batch_label[np.newaxis, ...], dtype=tf.int32)

            pred, _ = model(batch_data, training=False, bn_decay=1.0)
            pred = tf.argmax(pred, axis=2).numpy()[0]

            correct = np.sum(pred == batch_label.numpy()[0])
            accs.append(correct / NUM_POINT)
            preds.append(pred)
            labels.append(batch_label.numpy()[0])

    all_pred = np.concatenate(preds, axis=0)
    all_label = np.concatenate(labels, axis=0)
    ious, mIoU = compute_iou(all_pred, all_label, NUM_CLASSES)
    acc = np.mean(accs)

    log_string(f'整体准确率: {acc:.4f}')
    log_string(f'每个类别的IoU:')
    for cls_idx in range(NUM_CLASSES):
        log_string(f'  {CLASS_NAMES[cls_idx]}: {ious[cls_idx]:.4f}')
    log_string(f'平均mIoU: {mIoU:.4f}')

    with summary_writer.as_default():
        tf.summary.scalar(f'{"whole_scene_" if is_whole_scene else ""}accuracy', acc, step=epoch)
        tf.summary.scalar(f'{"whole_scene_" if is_whole_scene else ""}mIoU', mIoU, step=epoch)

    return acc, ious, mIoU


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