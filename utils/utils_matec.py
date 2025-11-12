

# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta




def build_dataset(config):
    """
    构建数据集：加载数据 + 总长度对齐（截断/填充）+ 归一化
    :param config: 配置类实例
    :return: 处理后的数据集（list of (对齐特征, 标签)）
    """
    # 1. 定义tokenizer：按空格拆分特征（你的数据是空格分隔）
    tokenizer = lambda x: x.split(' ')
    
    def load_dataset(path):
        contents = []
        # 统计长度处理情况（可选，方便排查）
        truncate_count = 0  # 被截断的样本数
        pad_count = 0       # 被填充的样本数
        normal_count = 0    # 长度刚好的样本数
        
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f, desc=f"加载 {os.path.basename(path)}"):
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                
                # 2. 拆分“特征序列”和“标签”（你的数据用\t分隔）
                try:
                    content_str, label_str = line.split('\t')
                except ValueError:
                    print(f"跳过格式异常行：{line}")  # 避免因行格式错误崩溃
                    continue
                
                # 3. 特征转换：字符串→float（并过滤空字符串，避免split后有无效元素）
                token = [float(x) for x in tokenizer(content_str) if x.strip()]
                
                # 4. 长度对齐：截断（超过总长度）/ 填充（不足总长度）
                current_len = len(token)
                if current_len > config.total_len:
                    # 超过总长度：截断到total_len（保留前N个特征，符合时序数据逻辑）
                    token_aligned = token[:config.total_len]
                    truncate_count += 1
                elif current_len < config.total_len:
                    # 不足总长度：用pad_value填充到total_len（尾部填充，网络数据常用）
                    token_aligned = token + [config.pad_value] * (config.total_len - current_len)
                    pad_count += 1
                else:
                    # 长度刚好：直接保留
                    token_aligned = token
                    normal_count += 1
                
                # 5. 特征归一化（论文要求：字节值0-255 → [0,1]，提升模型训练稳定性）
                if config.normalize:
                    token_aligned = [x / 255.0 for x in token_aligned]
                
                # 6. 加入结果（特征+标签）
                contents.append((token_aligned, int(label_str)))
        
        # 打印长度处理统计（可选，方便了解数据情况）
        print(f"数据统计 - 总样本数：{len(contents)} | 截断数：{truncate_count} | 填充数：{pad_count} | 长度刚好数：{normal_count}")
        return contents

    train = load_dataset(config.train_path)
   
    dev = load_dataset(config.dev_path)
    test = load_dataset(config.test_path)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
      
        p1 = torch.FloatTensor([_[0] for _ in datas]).to(self.device)
        #p2 = torch.FloatTensor([_[1] for _ in datas]).to(self.device)
        #p3 = torch.FloatTensor([_[2] for _ in datas]).to(self.device) 
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        
        p1 = torch.reshape(p1,(p1.shape[0],3,786))
        #
        #p2 = torch.reshape(p2,(p2.shape[0],1,786))
        #p3 = torch.reshape(p3,(p3.shape[0],1,786))
        
        return p1,y
        
    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

