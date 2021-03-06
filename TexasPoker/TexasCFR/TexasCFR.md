# 基于虚拟遗憾最小化（CFR）的两人有限注德州扑克（Hold‘em）

​		以德州扑克作为实验的研究对象，利用虚拟遗憾最小化算法不断调整学习最优策略，最终实现一个具有较高智能水平的德州扑克博弈系统。

## Schedule

* 1.27 - 2.29： 文献翻译，开题报告，
* 3.1-3.21：两人Kuhn扑克CFR，Hold'em 基础框架 

## Completion

* 阅读文献理解CFR算法的思路和发展历程，通过练习基于CFR的两人Kuhn扑克熟悉算法，尝试远程服务器。

* 确定整体框架，所包含的类，游戏对局，牌桌，玩家，扑克牌，荷官，动作，策略，牌级（摊牌后手牌与公共牌的最大牌力组合），牌力（两张手牌）等。

* 玩家(Player)类：包含编号(id)，筹码(stack)，手牌(hands)，动作集合(actions)，是否为人类玩家(is_human)，大小盲注(blind_type)等元素，编号和筹码初始化时设定，手牌，动作采取，盲注类型以及筹码盈亏都由内部函数设定，同时还自带显示信息，动作采取等函数。

* 扑克牌(Card)类：包含数字(number)和花色(suit)两个属性，并带有一些检查的函数，以及快速判断与另一张牌是否为同一花色或同一数字。

* 动作(Action)类：只有一个序号(index)属性，根据序号确定是看牌(check)，下注(bet)，跟注(call)，加注(raise)，弃牌(fold)中的哪一个动作。由于玩家在一轮游戏中可采取的动作与上一个玩家的动作有很大关系，所以定义三个策略集合，如上一个玩家下注，那下一个玩家只能采取跟注，加注或弃牌；如果上一个玩家看牌，拿下一个玩家只能选择看牌，下注或弃牌等等。根据玩家的历史动作集合以及游戏状态：翻牌前(pre-flag)，翻牌(flag)，转牌(turn)，河牌(river)，便可确定唯一的可采取动作集合供玩家选择，这个过程在玩家类的动作采取函数中完成。

* 牌桌(Deck)类：包含52张扑克牌，在游戏开始由荷官创建，具有洗牌，发牌等函数。

* 荷官(Dealer)类：包含牌桌，钱罐(pot)，公共牌等属性，对牌桌进行操作，在不同游戏状态执行不同的函数。

* 牌级(Rank)类：摊牌后，两张玩家手牌与五张公共牌组合最大牌级。将最大组合牌级以数组形式返回，如下：

  ```
  Royal-flush? (9, 0, 0, 0, 0)
  Straight-flush? (8, highest card, 0, 0, 0, 0)
  4-of-a-kind? (7, card in 4s, kicker, 0, 0, 0)
  Full house? (6, card in 3s, card in 2s, 0, 0, 0)
  Flush? (5, highest card, 2nd, 3rd, 4th, 5th)
  Straight? (4, highest card, 0, 0, 0, 0)
  3-of-a-kind? (3, card in 3s, kicker, 2nd, 0, 0)
  2 pairs? (2, high pair, low pair, kicker, 0, 0)
  Pair? (1, pair, kicker, 2nd, 3rd, 0)
  Nothing? (0, 1st, 2nd, 3rd, 4th, 5th)
  ```

* 游戏(Game)类：在未加入CFR策略时用玩家-玩家对战测试程序，无误后再引入CFR算法计算游戏策略并训练高智能体。首先创建一局游戏，玩家个数默认为2，给每人分配一定的筹码，并分别下大小盲注；创建荷官和牌桌并生成52张扑克牌，随机洗牌后，荷官给两名玩家发两张手牌并出示三张公共牌，根据玩家不同的对局状态（翻牌前，翻牌，转牌，河牌）预先设置好可选择的动作集合（看牌，下注，跟注，加注，弃牌），玩家在不同阶段选择。如果有一人选择弃牌，则另一名玩家获胜。如果没有人弃牌，则河牌后比较两人牌力大小，牌力大者获胜。

## Plan

* 尽快实现CFR，从抽象方法等方面进行改动。

  