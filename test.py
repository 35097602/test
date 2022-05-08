# r为目标wh和锚框wh的比值，比值在0.25到4之间的则采用该种锚框预测目标
r = t[:, :, 4:6] / anchors[:, None]  # wh ratio：计算标签box和当前层的anchors的宽高比，即:wb/wa,hb/ha
# 将比值和预先设置的比例anchor_t对比，符合条件为True，反之False
j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
