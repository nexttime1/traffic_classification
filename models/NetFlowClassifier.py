import torch
import torch.nn as nn
import torch.nn.functional as F



class Config(object):
    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'NetFlowClassifier'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]
        self.vocab_path = dataset + '/data/vocab.pkl'
        self.save_path = './datasets/NetFlowClassifier/app/saved_dict/NetFlowClassifier.ckpt'  # 模型保存路径
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 数据与模型结构参数
        self.num_classes = len(self.class_list)
        self.n_vocab = 0
        self.pad_size = 786
        self.embed = 432
        self.dim_model = 432
        self.packet_len = 786
        self.n_packets = 3
        self.total_len = self.packet_len * self.n_packets  # 3×786=2358
        self.pad_value = 0.0
        self.normalize = True

        # 训练超参数
        self.num_epochs = 150
        self.learning_rate = 0.001
        self.batch_size = 64
        self.dropout = 0.2
        self.require_improvement = 2000
        self.num_encoder = 3
        self.num_head = 4  # 432/4=108 每个头108维


# ===========================
# === Attention Components ===
# ===========================

class Scaled_Dot_Product_Attention(nn.Module):
    """Scaled Dot-Product Attention"""
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    """多头注意力机制"""
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        assert dim_model % num_head == 0
        self.num_head = num_head
        self.dim_head = dim_model // num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale)
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        return out


# ===========================
# === Add & Norm ===
# ===========================

class Add_Norm(nn.Module):
    """标准 Transformer 残差归一化模块"""
    def __init__(self, dim_model, dropout=0.1):
        super(Add_Norm, self).__init__()
        self.layer_norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, origin_x, sublayer_out):
        out = self.dropout(sublayer_out)
        out = self.layer_norm(origin_x + out)
        return out


# ===========================
# === Feed Forward ===
# ===========================

class Feed_Forward(nn.Module):
    """前馈全连接层"""
    def __init__(self, dim_model, hidden_dim=1024):
        super(Feed_Forward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_model, hidden_dim),
            nn.GELU(),             #
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, dim_model)
        )

    def forward(self, x):
        return self.fc(x)


# ===========================
# === Encoder Layer ===
# ===========================

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Feed_Forward(dim_model)
        self.add_norm1 = Add_Norm(dim_model, dropout)
        self.add_norm2 = Add_Norm(dim_model, dropout)

    def forward(self, x):
        attn_out = self.attention(x)
        out = self.add_norm1(x, attn_out)
        ff_out = self.feed_forward(out)
        out = self.add_norm2(out, ff_out)
        return out


# ===========================
# === Main Model ===
# ===========================

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 1D卷积特征提取层
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        )
        # 线性映射到Transformer输入维度
        self.fc1 = nn.Linear(config.pad_size, config.dim_model)

        # 堆叠Encoder层
        self.encoder1 = Encoder(config.dim_model, config.num_head, config.dropout)
        self.encoder2 = Encoder(config.dim_model, config.num_head, config.dropout)

        # Flatten后分类层
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1296, config.num_classes)  # 3×432=1296

    def forward(self, x):
        # 输入形状 (batch, 3, 786)
        x = self.conv1d(x)
        # Conv1d输出形状 (batch, 3, 786)
        out = self.fc1(x)
        out = self.encoder1(out)
        out = self.encoder2(out)
        out = torch.flatten(out, start_dim=1)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

