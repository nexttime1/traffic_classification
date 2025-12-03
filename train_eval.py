# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils.utils_netFlowClassifier import get_time_dif
import os


def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):

    # 初始化带标签平滑的损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑参数设置为0.1
    
    start_time = time.perf_counter()
    model.train()
    # 优化器添加L2正则化（weight_decay）
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=1e-5  # L2正则化系数，减轻过拟合
    )
    
    # 添加学习率调度器：每30个epoch学习率乘以0.5
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=30,  # 每30个epoch调整一次
        gamma=0.5      # 衰减系数
    )

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    save_dir = os.path.dirname(config.save_path)  # 提取保存目录
    os.makedirs(save_dir, exist_ok=True)  # 自动创建目录

    for epoch in range(config.num_epochs): 
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.8f}")  # 打印当前学习率
        
        for i, (traffic, labels) in enumerate(train_iter): 
            preds = model(traffic)
            loss = criterion(preds, labels)  # 使用带标签平滑的损失函数
           
            optimizer.zero_grad()               
            loss.backward()       
            optimizer.step()
          
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(preds.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                
                # 验证集loss下降时保存模型
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                
                model.train()
                
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, {5}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, improve))
             
            total_batch += 1
            # 早停逻辑：验证集loss超过200000 batch没下降则停止
            if total_batch - last_improve > 200000:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        
        # 每个epoch结束后更新学习率
        scheduler.step()
        
        if flag:
            break
    
    end_time = time.perf_counter()
    print(f"Training time usage: {end_time - start_time:.2f} seconds")
    test(config, model, test_iter)


def test(config, model, test_iter):
    
    if not os.path.exists(config.save_path):
        raise FileNotFoundError(f"模型文件不存在！路径：{config.save_path}\n请确认训练时模型已成功保存。")
    
    # 加载模型并测试
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    
    # 打印测试结果
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Test time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    start_time = time.time()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
   
    with torch.no_grad():  # 评估时禁用梯度计算，节省内存和时间
        for traffic, labels in data_iter:
            outputs = model(traffic)
            loss = F.cross_entropy(outputs, labels)  # 评估时用普通损失函数
            loss_total += loss.item()  # 用.item()避免张量累积，节省内存
            
            # 提取标签和预测结果
            labels_np = labels.data.cpu().numpy()
            predic_np = torch.max(outputs.data, 1)[1].cpu().numpy()
            
            # 拼接所有样本结果
            labels_all = np.append(labels_all, labels_np)
            predict_all = np.append(predict_all, predic_np)

    time_dif = get_time_dif(start_time)
    if test:
        print(f"Evaluation time usage: {time_dif}")
    
    # 计算准确率
    acc = metrics.accuracy_score(labels_all, predict_all)
    # 测试时额外返回分类报告和混淆矩阵
    if test:
        report = metrics.classification_report(
            labels_all, predict_all, 
            target_names=config.class_list, 
            digits=4,  # 保留4位小数，结果更精确
            zero_division=0  # 避免某些类别无预测时报错
        )
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    
    return acc, loss_total / len(data_iter)
