import time

import torch
import wandb
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
from LemonLeavesDiseaseModel import LemonLeavesDiseasesModel
import numpy as np


def lemon_leaves_train_model(model, train_loader, val_loader, optimizer, criterion, epochs, classes_types,
                             device='cpu', model_suffix="LemonLeaves") -> LemonLeavesDiseasesModel:
    __init_wandb("Lemon Leaves Disease Classification", epochs, batch_size=train_loader.batch_size)

    classes_num = len(classes_types)
    # 定义性能指标
    train_metrics = __get_metrics_collector(classes_num).to(device)
    val_metrics = __get_metrics_collector(classes_num).to(device)

    # 最小验证集损失
    best_val_loss = float('inf')

    # 开始训练
    train_loader_len = len(train_loader)
    val_loader_len = len(val_loader)
    for epoch in range(epochs):
        print(f"Start {epoch + 1}/{epochs} Epoch......")
        model.train()
        train_losses = []
        batch_id = 0
        for inputs, labels in train_loader:
            cur_time = time.time()
            print(f"Start Train Batch {batch_id + 1}/{train_loader_len}")
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            # 记录损失
            train_losses.append(loss.item())

            # 更新指标
            train_metrics.update(outputs, labels)

            # 更新批次索引
            print(f"Train Batch {batch_id + 1}/{train_loader_len} End. Time Using: {time.time() - cur_time:.4f}")
            batch_id += 1

        # 计算验证集预测结果
        model.eval()
        val_losses = []
        val_probs = []  # 概率向量（每个样本的各类别概率）
        val_targets = []  # 真实标签
        batch_id = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                cur_time = time.time()
                print(f"Start Val Batch{batch_id + 1}/{val_loader_len}")
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                val_losses.append(loss.item())

                # 更新指标
                val_metrics.update(outputs, labels)

                # 收集预测标签和概率
                probs = torch.softmax(outputs, dim=1)

                # 批量保存
                val_probs.append(probs.cpu())
                val_targets.append(labels.cpu())

                # 更新批次索引

                print(f"Val Batch {batch_id + 1}/{val_loader_len} End. Time Using: {time.time() - cur_time:.4f}")
                batch_id += 1

        # 合并所有批次的结果
        val_probs = torch.cat(val_probs).numpy()
        val_targets = torch.cat(val_targets).numpy()

        # 计算损失与性能指标
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        __upload_wandb(train_loss, train_metrics.compute(), "Train")
        __upload_wandb(val_loss, val_metrics.compute(), "Val", probs=val_probs, labels=val_targets,
                       classes_types=classes_types)
        wandb.log({})

        # 重置性能指标
        train_metrics.reset()
        val_metrics.reset()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Best Model update! The Best Val Loss is {best_val_loss}")
            torch.save(model.state_dict(), f"../ModelWeights/BestModel-{model_suffix}.pth")

    wandb.finish()
    return model


def __get_metrics_collector(classes_num: int) -> MetricCollection:
    metrics = {"acc": Accuracy(task="multiclass", num_classes=classes_num),
               "precision": Precision(task="multiclass", num_classes=classes_num, average="macro"),
               "recall": Recall(task="multiclass", num_classes=classes_num, average="macro"),
               "f1": F1Score(task="multiclass", num_classes=classes_num, average="macro"),
               }
    return MetricCollection(metrics)


def __init_wandb(project_name, num_epochs, batch_size, optimizer_type='AdamW', weight_decay=0):
    # 初始化WandB
    wandb.init(project=project_name,dir="../WandBLogs/")
    wandb.config.update({
        'epochs': num_epochs,
        'batch_size': batch_size,
        'weight_decay': weight_decay,

        'model_name': 'efficientnet_v2_m',
        'model_weights': 'DEFAULT',
        'optimizer': optimizer_type,
    })


def __upload_wandb(loss: np.floating, metrics: dict, prefix: str, probs=None, labels=None,
                   classes_types=None):
    __upload_wandb_loss_data(loss, prefix)

    __upload_wandb_metrics_stats(metrics, prefix)

    if probs is not None and labels is not None and classes_types is not None:
        __upload_val_stats_table(probs, labels, classes_types, prefix)


def __upload_wandb_loss_data(loss: np.floating, prefix: str):
    wandb.log({f'{prefix} Loss': loss, }, commit=False)


def __upload_wandb_metrics_stats(metrics: dict, prefix):
    # 上传性能指标
    wandb.log({
        # 准确率
        f'{prefix} Accuracy': metrics["acc"],
        # 精确率
        f'{prefix} Precision': metrics["precision"],
        # 召回率
        f'{prefix} Recall': metrics["recall"],
        # F1值
        f'{prefix} F1': metrics["f1"],
    },
        commit=False
    )


def __upload_val_stats_table(probs, labels, classes_names: list, prefix):
    wandb.log({
        # 混淆矩阵
        f'{prefix} CM': wandb.plot.confusion_matrix(probs=probs, y_true=labels, class_names=classes_names),
        # ROC曲线
        f'{prefix} ROC': wandb.plot.roc_curve(labels, probs, labels=classes_names)
    },
        commit=False)
