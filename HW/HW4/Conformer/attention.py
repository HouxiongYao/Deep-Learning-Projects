import  torch
import torch.nn as  nn
import torch.nn.functional as F
class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        # 初始化多头注意力的参数
        self.n_head = n_head  # 头数
        self.d_model = d_model  # 模型维度
        self.d_head = d_head  # 每个头的维度
        self.dropout = dropout  # dropout的概率

        # 查询、键、值的线性变换层
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        # Dropout层
        self.drop = nn.Dropout(dropout)  # 输出的dropout
        self.dropatt = nn.Dropout(dropatt)  # 注意力分数的dropout
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)  # 输出的线性变换层

        # 层归一化层
        self.layer_norm = nn.LayerNorm(d_model)

        # 缩放因子
        self.scale = 1 / (d_head ** 0.5)

        # 是否在多头注意力前应用层归一化
        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        """
        创建一个平行四边形形状的mask，用于限制注意力矩阵的视野。
        """
        mask = torch.ones((h, w)).byte()  # 初始化全1的mask
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])  # 上三角
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])  # 下三角

        if left:
            return mask
        else:
            return mask.flip(0)  # 如果不是左侧，就将mask反转

    def _shift(self, x, qlen, klen, mask, left=False):
        """
        将输入张量x做shift操作（加零填充并按需翻转），用于处理相对位置编码。
        """
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)  # 填充零

        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)  # 如果是左shift，翻转mask
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)  # 拼接并扩展
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)  # 拼接并扩展

        x = x_padded.masked_select(mask[:, :, None, None]) \
                    .view(qlen, klen, x.size(2), x.size(3))  # 根据mask选择元素

        return x

    def _rel_shift(self, x, zero_triu=False):
        """
        执行相对位置的shift操作，用于生成相对位置编码。
        """
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)  # 创建零填充

        x_padded = torch.cat([zero_pad, x], dim=1)  # 拼接零填充和输入x

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])  # 调整维度

        x = x_padded[1:].view_as(x)  # 除去第一个元素

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))  # 创建全1的张量
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]  # 只保留下三角部分

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        # 计算前向传播。此方法在子类中实现。
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        # 添加一个新的线性层，用于计算r的相对位置编码
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        # 获取输入序列长度和batch size
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            # 如果mems存在，将mems与w拼接，并计算查询、键、值
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            # 将查询、键、值分开
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]  # 只保留w的最后一部分
        else:
            # 如果没有mems，直接计算查询、键、值
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            # 将查询、键、值分开
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        # klen表示键的长度
        klen = w_head_k.size(0)

        # 将查询、键、值的维度重塑成适合后续计算的形状
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        # 将r的键重塑为适合后续计算的形状
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)

        #### 计算注意力得分
        rw_head_q = w_head_q + r_w_bias  # 加上偏置
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # 计算查询和键的点积

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # 计算查询和r键的点积
        BD = self._rel_shift(BD)  # 执行相对位置偏移

        # 合并AC和BD得到最终的注意力得分
        attn_score = AC + BD
        attn_score.mul_(self.scale)  # 缩放

        #### 计算注意力概率
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)

        # 使用softmax计算注意力概率
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### 计算注意力向量
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # 将注意力向量reshape为合适的形状
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### 线性变换
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        # 添加残差连接并进行层归一化
        if self.pre_lnorm:
            output = w + attn_out
        else:
            output = self.layer_norm(w + attn_out)

        return output

if __name__=="__main__":
    n_head = 8  # 头数
    d_model = 512  # 模型维度
    d_head = 64  # 每个头的维度
    dropout = 0.1  # dropout 比例
    dropatt = 0.1  # attention dropout 比例
    batch_size = 2  # batch 大小
    qlen = 10  # query 序列长度
    rlen = 10  # key/value 序列长度
    w = torch.randn(qlen, batch_size, d_model)  # query sequence
    r = torch.randn(rlen,d_model)  # key/value sequence
    r_w_bias = torch.randn(n_head, d_head)  # relative position bias for w
    r_r_bias = torch.randn(n_head, d_head)  # relative position bias for r

    # 测试的掩码
    attn_mask = torch.zeros(qlen, rlen).byte()  # 假设没有掩码

    # 初始化模型
    attn_layer = RelPartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout)

    # 运行前向传播
    output = attn_layer(w, r, r_w_bias, r_r_bias, attn_mask)

    # 输出形状检查
    print("Output shape:", output.shape)