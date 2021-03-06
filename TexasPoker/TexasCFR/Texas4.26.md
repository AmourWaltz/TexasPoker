# 基于虚拟遗憾最小化（CFR）的两人有限注德州扑克（Hold‘em）

​		以德州扑克作为实验的研究对象，利用虚拟遗憾最小化算法不断调整学习最优策略，最终实现一个具有较高智能水平的德州扑克博弈系统。

## Schedule：4.14-4.26

* 改进存储方式，优化结构；
* 测试程序性能：空间，时间；
* 改进手牌加公共牌的牌级判别方式；
* 测试改进的9-bucketing策略。

## Details

* 之前将所有信息集存储在同一个文件，内存消耗较少（22.4MB），但是在迭代更新时需要消耗较多的时间遍历找到对应的节点（第一次生成需要16s，之后每次迭代一遍大约10min）；改为pickel的存储方式，并且将涉及到字符的信息全部用数字代替，然后将一个节点的信息用列表表示，每个信息集单独建立一个文件，最后内存消耗622MB，但迭代运行时间只有约20s，一种空间换时间的思路。

* 迭代了500次训练了一个每轮最多三次加注的程序，在与纯策略（全部选择叫注，无加注弃牌）的对局中，表现尚可，策略选取较保守，弃牌次数略多，不弃牌的情况一般都能取胜；由于训练中也有纯策略的情况，个人感觉是训练次数不够。

* 改进的9-bucketing策略，每个节点包含的不再是当前两张手牌的牌力，而是手牌与公共牌的最大组合，共有2,598,960种情况，均分到9个桶里，这样根据场上不同阶段公共牌的变化动态调整策略，也更符合实际情况；为适应这种策略改进了牌级判断方式，可以输入5-7张牌，返回最大组合的牌级。

* 简单测试了可行性，根据分析信息集不会增加，运行时间也不会增加太多，未来可期。

  


## Problems and optimization

* 虽然使用了抽象方法，但每次迭代依旧花费较长时间，开启四核运行后发热依旧很严重；下一步打算租用服务器。

