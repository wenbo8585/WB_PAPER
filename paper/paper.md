## Retrieval Metric

设$\{x_1，... x_n，...，x_N\}$
是一组图像的特征集合。 要查询图像$q$与数据库每个图像$x_n$的相似度，使用一个恰当的距离函数$d(q，x_n）$计算出$x_{n_i}$与$x_{n_{i+1}}$距离序列$\{ x_{n_1},...x_{n_i},...,x_{n_{i+1}}\}$。 
根据它们距离大小，例如$d（q，x_{n_i}）≤d（q，x_{n _{i+ 1}}）$对数据库图像进行排序。常用的距离函数有欧氏距离(Euclidean Distance)，余弦距离，马氏距离(Mahalanobis Distance)等。

本文使用precision-recall，average precision $AP$ 和平均精度（mean average precision, mAP）来评估CBIR的性能：

$$P = \frac{Number\ of\ relevant\ images\ retrieved}{Total\ number\ of\ images\ retrieved}$$

$$R = \frac{Number\ of\ relevant\ images\ retrieved}{Total\ number\ of\ relevant\ images}$$

给定一张待查询图像和返回列表，可以根据P-R值绘制P-R曲线。

$AP$ 是每个查询检索到的相关项的精度分数平均值（ The average precision AP for a single query q is the mean over the precision scores after each
retrieved relevant item）：

$$AP(q) = \frac{1}{N_R}\sum_{n=1}^{N_R}P_q(R_n),$$

$R_n$是检索到底$n$个相关图像的召回，$N_R$是待查询图像总数。它相当于精准度-召回率曲线下的面积。通常，较大的AP值意味着更高的精准度-召回率曲线，亦即更好的检索性能。

平均精度$mAP$是所有$AP$的平均值：

$$mAP = \frac{1}{|Q|}\sum_{q\in Q}AP(q),$$

$Q$ 是查询图像集合。

## Loss Function

loss function使用交叉熵去拟合one hot的分布,交叉熵的公式是:

$$S(q|p)=-\sum_i q_i \log p_i$$

​			$$S(q|p)=-\sum_i q_i \log p_i$$

其中$p_i​$是预测的分布，而$q_i​$是真实的分布.考虑到训练数据不平衡，相似度十分大的影响，训练loss function使用了加权代价交叉熵函数：

$$loss = q∗−\log(sigmoid(p*\theta))∗w+(1−q)∗−\log(1−sigmoid(p*\theta))$$

其中$w$是正样本系数，$\theta$是代价系数.

