# Offsets
# 得到相对于以左上角为坐标原点的坐标
gxy = t[:, 2:4]  # grid xy
# 得到相对于右下角为坐标原点的坐标
gxi = gain[[2, 3]] - gxy  # inverse
# 这两个条件可以用来选择靠近的两个邻居网格
# jk和lm是判断gxy的中心点更偏向哪里
j, k = ((gxy % 1 < g) & (gxy > 1)).T
l, m = ((gxi % 1 < g) & (gxi > 1)).T
j = torch.stack((torch.ones_like(j), j, k, l, m))
# yolov5不仅用目标中心点所在的网格预测该目标，还采用了距目标中心点的最近两个网格
# 所以有五种情况，网格本身，上下左右，这就是repeat函数第一个参数为5的原因
t = t.repeat((5, 1, 1))[j]
# 这里将t复制5个，然后使用j来过滤
# 第一个t是保留所有的gtbox，因为上一步里面增加了一个全为true的维度，
# 第二个t保留了靠近方格左边的gtbox，
# 第三个t保留了靠近方格上方的gtbox，
# 第四个t保留了靠近方格右边的gtbox，
# 第五个t保留了靠近方格下边的gtbox，
offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
