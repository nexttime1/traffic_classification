
import os
import random
# from utils import ID_TO_APP, ID_TO_TRAFFIC  # 只保留用到的导入  
from utils import ID_TO_TorAPP, ID_TO_TRAFFIC  #这个是tor
def splitDeepPacket(inpath, outpath):
    # 确保输出目录存在
    os.makedirs(outpath, exist_ok=True)
    
    # 读取预处理后的TXT文件
    with open(inpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    count = len(lines)
    if count == 0:
        print(f"错误：{inpath} 文件为空，请检查预处理是否成功")
        return
    
    # 随机打乱数据
    random.shuffle(lines)
    
    # 生成类别文件（class.txt）
   #  with open(os.path.join(outpath, "class.txt"), 'w', encoding='utf-8') as f_class:  
       #  if "app" in outpath:
            # 应用类型标签（如果需要）
       #      for i in ID_TO_APP:
        #         f_class.write(f"{ID_TO_APP[i]}\n")
      #   elif "service" in outpath:
      #       # 服务类型标签（MATEC主要用这个）
        #     for i in ID_TO_TRAFFIC:
        #         f_class.write(f"{ID_TO_TRAFFIC[i]}\n")
    
    
    # 
    with open(os.path.join(outpath, "class.txt"), 'w', encoding='utf-8') as f_class:
      if "app" in outpath:
        for i in ID_TO_TorAPP:  
            f_class.write(f"{ID_TO_TorAPP[i]}\n")
      elif "service" in outpath:
        for i in ID_TO_TRAFFIC:
            f_class.write(f"{ID_TO_TRAFFIC[i]}\n")
    
    
    
    # 划分比例：8:1:1（训练:验证:测试）
    train_split = int(count * 0.8)
    dev_split = int(count * 0.9)
    
    # 写入训练集、验证集、测试集
    with open(os.path.join(outpath, "train.txt"), 'w', encoding='utf-8') as f_train:
        f_train.writelines(lines[:train_split])
    
    with open(os.path.join(outpath, "dev.txt"), 'w', encoding='utf-8') as f_dev:
        f_dev.writelines(lines[train_split:dev_split])
    
    with open(os.path.join(outpath, "test.txt"), 'w', encoding='utf-8') as f_test:
        f_test.writelines(lines[dev_split:])
    
    print(f"成功分割数据：训练集{train_split}条，验证集{dev_split-train_split}条，测试集{count-dev_split}条")
    print(f"文件保存至：{outpath}")

if __name__ == '__main__':
    # 预处理输出路径（dpProcess_vpn.py生成的datanet_service.txt）
    # inpath_service = "./service/datanet_service.txt"  # 预处理后service文件夹下的TXT
    # 保存训练/测试集的路径
    # outpath_service = "../datasets/NetFlowClassifier/service/data"  # 确保上级目录存在
    
    inpath_app = "./app/datanet_app.txt"
    # 输出：TOR数据集的训练/验证/测试集存放路径
    outpath_app = "../datasets/NetFlowClassifier/app/data"
    
    
    
    # 执行分割
    # splitDeepPacket(inpath_service, outpath_service)
    # tor 的分割
    splitDeepPacket(inpath_app, outpath_app)
