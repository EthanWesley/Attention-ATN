
import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(CrossModalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)

    def forward(self, x, y):
        # x和y的形状应为(seq_length, batch_size, dim)
        # 将x作为查询，y作为键和值
        attn_output_1, _ = self.attention(x, y, y)
        return attn_output_1


class AttentionFusionModel(nn.Module):
    def __init__(self, modal1_model, modal2_model, clinical_feature_dim, dropout=0.5, num_heads=4):
        super(AttentionFusionModel, self).__init__()

        self.modal1_model = modal1_model
        self.modal2_model = modal2_model

        # 获取单模态模型的输出特征维度
        modal1_output_dim = 512
        modal2_output_dim = 512

        # 定义跨模态注意力层
        self.cross_modal_attention = CrossModalAttention(modal1_output_dim, num_heads)

        # 创建一个全连接层用于融合两个模态的特征
        self.fusion_fc = nn.Linear(modal1_output_dim + modal2_output_dim + clinical_feature_dim, 2)
        # 这里的num_classes是你任务中的类别数
        self.dropout = nn.Dropout(dropout)

    def forward(self, modal1_input, modal2_input, clinical_features):
        # 前向传播过程中将两个模态的输入分别传递给对应的单模态模型
        modal1_output = self.modal1_model(modal1_input)
        modal2_output = self.modal2_model(modal2_input)

        # 应用跨模态注意力
        # 需要确保modal1_output和modal2_output的维度与跨模态注意力层匹配
        # modal1_output = modal1_output.unsqueeze(0)  # 增加seq_length维度
        # modal2_output = modal2_output.unsqueeze(0)
        attn_output_1 = self.cross_modal_attention(modal1_output, modal2_output)
        # attn_output_1 = attn_output_1.squeeze(0)  # 移除seq_length维度

        # 使用注意力权重调整模态1的输出特征
        modal1_output_weighted = attn_output_1 * modal1_output
       

        attn_output_2 = self.cross_modal_attention(modal2_output, modal1_output)
        # attn_output_2 = attn_output_2.squeeze(0)  # 移除seq_length维度

        # 使用注意力权重调整模态1的输出特征
        modal2_output_weighted = attn_output_2 * modal2_output
        # # 将两个模态的特征进行拼接
        # fused_features = torch.cat((modal1_output, modal2_output, clinical_features), dim=1)

        # 将两个模态的特征进行拼接
        fused_features = torch.cat((modal1_output_weighted, modal2_output_weighted, clinical_features), dim=1)

        # 使用全连接层融合特征并进行最终的分类
        fused_features = self.dropout(fused_features)
        output = self.fusion_fc(fused_features)

        return output




        # 维度匹配：确保modal1_output和modal2_output的维度与跨模态注意力层所需的维度相匹配。
        # 如果模型输出是多维的，您可能需要相应地调整维度。
        # 超参数：注意力层的超参数（如num_heads）需要根据模型和数据进行调整。确保注意力层的输入维度dim与模态模型的输出特征维度一致。
        # 性能调优：加入跨模态注意力后，可能需要重新调整模型的其他部分或训练过程中的超参数，以达到最佳性能。