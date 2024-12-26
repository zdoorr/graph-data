# 复旦大学2024年秋季学期图数据管理与挖掘期末Project

## 团队成员
- DRZ, LZK

## 背景
- 基于蒙特卡洛模拟和矩阵计算，开发一系列用于图同构检验问题的剪枝算法，在执行具体的同构检查算法（如VF2算法）之前，迅速筛选掉不合理的图。
- 具体的场景可能是：数据库中已有大量已知的图数据（如化学分子式），给定查询图（q），尽可能快速地查找到数据库中所有与q同构的图。数据库中的图为半静态数据，更新较慢，能够通过离线计算的方式挖掘其各类性质用于剪枝。查询图为流式数据，其结构信息难以提前知道。

## 文件目录树
- **请DRZ同学协助完成，将项目文件结构整理成如下目录树**，并保持代码注释风格一致，可参考[utils.py](./utils.py)文件
```
|-- graph-data
    |-- graphs  # 存放图数据，格式为pkl
    |-- image # 存放图片
    |-- .gitignore # git忽略文件，为节省存储空间，忽略pkl和图片文件的更新，因为可以在本地生成
    |-- readme.md # 项目说明文档
    |-- eigenvalue_research.py # 研究不同种类的邻居矩阵特征值分布
    |-- Monte_Carlo_prune.py # 使用蒙特卡洛模拟进行剪枝
    |-- Lancoz_prune.py # 使用Lancoz算法进行剪枝
    |-- Perron_multiplicity_prune.py # 估计Perron根的重数，用于剪枝
    |-- utils.py # 存放工具函数
```

## 任务目标
- [x] 邻接矩阵特征值性质研究。目前发现带权重、对角元非零（无周期）的邻接矩阵具有较好的特征值性质。此外，行归一化的矩阵（也即马尔可夫转移矩阵）、对称矩阵（对应无向图）也都有较好的特征值性质。以上发现可以为后续的蒙特卡洛模拟、Lancoz算法提供一些启发。
- [ ] 蒙特卡洛模拟
  - 用degree作为权重，**效果很好**，后续优化方向：0权重节点预处理、初始粒子分布改进、尝试使用其他关于degree的函数
  - （可以尝试）用label作为权重，比如化学原子的质量数、电荷数
  - 提前终止模拟的阈值尽量提高
  - 除了MSE，也可以考虑用JS散度，或者Wasserstein来度量两个归一化向量的距离。
  - 可以生成一些正例和负例样本，然后跑一下逻辑回归来寻找最佳的Threshold。
  - 生成随机图的时候，为了保持$|E|=O(V)$的稀疏性，请将边的生成概率控制在$p=O(1/|V|)$。
- [ ] Lancoz算法：可以很好地利用矩阵的稀疏性来加快计算。前期数值实验表明对称的、且对角线非零（相当于马尔科夫链无周期）的矩阵的特征值不容易出现重根，因此Lancoz算法结合Cauchy交错定理，应该也会有很好的剪枝效果。
- [ ] Perron根重数估计。来自邵美誉悦老师的一个建议。前期数值实验发现，带权重的、行归一化的邻接矩阵没有虚数根，而且Perron根（也即最大的特征根，为1）的重数往往比较高。Perron根的重数也可以用于剪枝。例如，做一个在1附近的围道积分：
$$\text{tr} \left(\frac{1}{2\pi\text{i}}\int_{\Gamma}1(\lambda I- A)^{-1} \text{d}\lambda\right)$$
可以估计出Perron根代数重数的上界。

## 后续任务
- [ ] Presentation准备（PPT制作、试讲），1月3日汇报
- [ ] 期末英文报告撰写，1月10日前提交

## 项目日志
- 2024.12.25 完成邻接矩阵特征值探索性实验。
- 2024.12.25 线下讨论，确定了尝试Perron根重数、蒙特卡洛模拟、Lancoz算法三种剪枝方式。对蒙特卡洛权重设置问题、相近图的蒙特卡洛粒子分布差异的度量问题、项目代码结构等进行了讨论。
- 2024.12.25 开始蒙特卡洛模拟实验，为蒙特卡洛模拟添加提前终止条件：粒子分布sort后归一化MSE小于阈值。开始模拟后，首先所有节点不设置权重，对相似无向图区分能力很差，对相似有向图，k=5时粒子分布出现一定差异，但MSE并无明确阈值。后发现权重可设置为：无向图\( d^2 \),有向图\( d_{\text{入}} \cdot d_{\text{出}} \)，灵感为经过节点的局部路径数量。对相似无向图（邻二甲苯、间二甲苯、对二甲苯解构）进行模拟，粒子分布差异明显。对相似有向图进行模拟，粒子分布差异观察可得，效果明显。
- 2024.12.26 项目文件目录树重构。
 
