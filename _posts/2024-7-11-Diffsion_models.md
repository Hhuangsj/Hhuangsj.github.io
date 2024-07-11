---
layout: post
title: "扩散模型"
date:   2024-7-11
tags: [tag1, tag2]
comments: true
author: hhuangsj
---


参考：
[https://segmentfault.com/a/1190000043744225#item-3-5](https://segmentfault.com/a/1190000043744225#item-3-5)
[https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
[https://yang-song.net/blog/2021/score/](https://yang-song.net/blog/2021/score/)

扩散模型由**前向过程**和**反向过程**这两部分组成:
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719839115972-03d46513-7685-4c83-a7c7-00b6917fd6e4.png#averageHue=%23f7f6f6&clientId=u38152ef9-498b-4&from=paste&height=180&id=ufeb3d4af&originHeight=270&originWidth=745&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=56691&status=done&style=none&taskId=u05aa7cde-dc77-4cad-a321-cbffdd8cf2e&title=&width=496.6666666666667)
(image source: [Ho et al. 2020](https://arxiv.org/abs/2006.11239))
## 前向过程
正向过程中，输入$x_0$会不断混入高斯噪声。经过𝑇次加噪声操作后，图像$𝑥_𝑇$会变成一幅符合标准正态分布的纯噪声图像。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719972078953-a03bac33-f243-4bbb-9173-3193f14cb363.png#averageHue=%23a8b39c&clientId=u603e3df7-f4cc-4&from=paste&height=251&id=ue4ad2151&originHeight=377&originWidth=1344&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=278782&status=done&style=none&taskId=u6842c987-d741-4110-8fc9-5c8ea25f22c&title=&width=896)
**公式表示：**
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719839509563-4c96fed1-24dc-4f54-80ee-5351db56c3db.png#averageHue=%23f5f5f5&clientId=u38152ef9-498b-4&from=paste&id=Uab8U&originHeight=71&originWidth=559&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=6547&status=done&style=none&taskId=u3a71a4df-c020-4d0d-9e3b-24d8511d964&title=)
前提：假设$x_{t-1}$符合正态分布
**拆解：**
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719841972953-ec3dd971-5dfe-4a6a-9582-61a221765846.png#averageHue=%23fefdfc&clientId=u38152ef9-498b-4&from=paste&id=u2aa195e5&originHeight=78&originWidth=431&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=8218&status=done&style=none&taskId=uce396765-ea7f-462b-9488-1de83500b1c&title=)
根据这个公式倒推1步：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719843971725-08b9bd15-8ad2-4570-bf1c-a38d6927ad5c.png#averageHue=%23fefdfc&clientId=u38152ef9-498b-4&from=paste&id=u46f232e8&originHeight=147&originWidth=573&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=12287&status=done&style=none&taskId=ue9540c3f-fbb0-4462-a37a-e071ac91676&title=)
由正态分布的性质可知，均值相同的正态分布“加”在一起后，**方差也会加到一起**。这样就能将后面两项合并成一个均值相同，方差相加的gaussian函数
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719845605712-e66a25b8-7855-4fb8-adb8-59a3c3f5c030.png#averageHue=%23ececec&clientId=u38152ef9-498b-4&from=paste&id=u970ed1b7&originHeight=63&originWidth=763&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=11495&status=done&style=none&taskId=uc580ea32-3477-4502-93c5-887edfb43a9&title=)
简化：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719844840392-402d641e-9df3-4506-b4f9-d189bc18fd91.png#averageHue=%23fefcfb&clientId=u38152ef9-498b-4&from=paste&id=u1c89af3d&originHeight=46&originWidth=250&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=3125&status=done&style=none&taskId=u486bf528-eb08-4aa7-a0ce-7e0ed7ca34b&title=)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719844847482-b600c334-b235-49fe-83a1-564c679d71b9.png#averageHue=%23fefdfc&clientId=u38152ef9-498b-4&from=paste&id=uef47295e&originHeight=55&originWidth=219&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=2627&status=done&style=none&taskId=u18f4205f-0a68-4cad-a535-9adca336a67&title=)
到这里前向过程的每一步加点噪声都可以知道
## 反向过程
反向过程中，我们希望训练出一个神经网络，它能够学会T个去噪声操作，把$𝑥_𝑇$还原回$x_0$。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719839115972-03d46513-7685-4c83-a7c7-00b6917fd6e4.png#averageHue=%23f7f6f6&clientId=u38152ef9-498b-4&from=paste&height=180&id=E9U4O&originHeight=270&originWidth=745&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=56691&status=done&style=none&taskId=u05aa7cde-dc77-4cad-a321-cbffdd8cf2e&title=&width=496.6666666666667)
网络的学习目标是让𝑇个去噪声操作**正好能抵消掉对应的加噪声操作**。训练完毕后，只需要从标准正态分布里随机采样出一个噪声，再利用反向过程里的神经网络把该噪声恢复成一幅图像，就能够生成一幅图片了。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719847296869-cac07aec-781c-4413-8054-579b3ce8e4aa.png#averageHue=%23f4f4f4&clientId=u38152ef9-498b-4&from=paste&id=u7ab7c7e8&originHeight=67&originWidth=627&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=7482&status=done&style=none&taskId=u7a3aab79-d4e8-487d-9f35-3bdf2b20b1f&title=)

**可以假设t-1步是在这样一个正态分布中采样：**
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719881507305-fc3f895d-316a-47aa-bc73-e60bc25681fd.png#averageHue=%23f8f6f6&clientId=u38152ef9-498b-4&from=paste&id=uf2ef1587&originHeight=50&originWidth=362&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=5129&status=done&style=none&taskId=ufd6eb3a0-3ff4-456b-a11a-800fde1d349&title=)
ps：在给定$x_0$的条件下，在xt情况下xt-1出现的概率
相当于拟合t-1步下的**均值和方差**（就是神经网络要干的事情）

通过贝叶斯公式得：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719881714296-594585ef-80fe-4d13-9116-065e788eb250.png#averageHue=%23fefcfb&clientId=u38152ef9-498b-4&from=paste&id=u6df6e646&originHeight=78&originWidth=391&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=5633&status=done&style=none&taskId=uf8ab0563-ee39-40be-a1bd-dc5f8a648a2&title=)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719882143537-fcf1ace8-126c-4f39-bfcc-b33b12350d0d.png#averageHue=%23fdfcfa&clientId=u38152ef9-498b-4&from=paste&id=u1f897d54&originHeight=189&originWidth=764&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=32656&status=done&style=none&taskId=u6cfd75a1-17fc-4f91-bd63-b4893a21dda&title=)
代入得：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719882164364-d608a1a0-06e6-431e-bda9-04082c64b1d5.png#averageHue=%23f7f6f6&clientId=u38152ef9-498b-4&from=paste&id=ufc464381&originHeight=232&originWidth=888&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=36066&status=done&style=none&taskId=u10d14c16-0463-4bf2-8c66-e5e55239060&title=)
最后得到分布的均值，方差分布为：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719882398471-a36502a7-54d9-431f-9f67-18bbbd57b78a.png#averageHue=%23fefdfc&clientId=u38152ef9-498b-4&from=paste&id=ua6f48313&originHeight=75&originWidth=259&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=3647&status=done&style=none&taskId=uc5f6fae5-a134-4e10-8d2f-8cb26cc65c9&title=)![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719882470056-0a7bf5af-5732-41b5-8845-19267ab0ff71.png#averageHue=%23fefdfc&clientId=u38152ef9-498b-4&from=paste&id=u87737ed8&originHeight=74&originWidth=181&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=2843&status=done&style=none&taskId=u7fea1a66-5d0c-4d63-963d-9424ff3ce15&title=)
𝛽𝑡是加噪声的方差，是一个常量。 逆过程也是一个常量。那么神经网络只用**拟合均值**
观察均值的公式可以发现，这里只有一个噪声𝜖𝑡是未知的。那神经网络干脆直接预测这个噪声ϵθ(xt,t)，让它和加噪过程对应的噪声ϵt的均方误差最小。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719883370765-48023347-f201-4b29-90d7-cf169f5abbb4.png#averageHue=%23fefdfc&clientId=u38152ef9-498b-4&from=paste&id=u0f28bbdd&originHeight=55&originWidth=226&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=2735&status=done&style=none&taskId=u49d141fe-5f82-4eca-b97c-d616c9a3741&title=)

以上只是一个简单的说法，是让**去噪声操作和加噪声操作的逆操**作尽可能相似。然而，这个对描述并不确切。扩散模型原本的目标，是**最大化pθ(x0)这个概率。换句话说：**给定一个训练集的数据**x**0，经过前向过程和反向过程，扩散模型要让**复原出x0的概率尽可能大**。
使用和VAE类似的变分推理，可以把优化目标转换成优化一个叫做**变分下界(variational lower bound, VLB)**的量。它最终可以写成：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719887385922-4f48ed5f-142c-4d46-8c5f-a7ff03ef8369.png#averageHue=%23fefcfb&clientId=u38152ef9-498b-4&from=paste&id=ua034eaef&originHeight=86&originWidth=763&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=10308&status=done&style=none&taskId=u5e005a98-40b0-401c-a7cf-595cf41301f&title=)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719887942619-c554fff7-a228-4a11-bb41-361b55b6be50.png#averageHue=%23fdfbf8&clientId=u38152ef9-498b-4&from=paste&id=uac444fe2&originHeight=164&originWidth=775&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=30978&status=done&style=none&taskId=u7c7bc796-1e2e-42c0-b6fb-9773554c4c3&title=)

![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719888335327-2ef7c7c1-c9e8-4c34-b935-14c6187ec330.png#averageHue=%23fefdfc&clientId=u38152ef9-498b-4&from=paste&id=u42f4afb5&originHeight=228&originWidth=771&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=20213&status=done&style=none&taskId=u968ae45f-6ae0-4718-b921-8ef308d2cdc&title=)

## 训练和采样算法
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1715224556662-e5b2cd3c-ef49-46fa-a49e-1efb8e009d13.png#averageHue=%23f5f4f4&clientId=uc7b1e96b-8990-4&from=paste&height=462&id=ufeea66e0&originHeight=700&originWidth=1054&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=196046&status=done&style=none&taskId=u869d9f43-79cf-47a4-a829-8e0e6f27331&title=&width=695)
### 训练
第二行：从训练集里取一个数据x0。
第三行：随机从1,...,T里取一个时刻用来训练。我们虽然要求神经网络拟合T个正态分布，但实际训练时，不用一轮预测T个结果，只需要随机预测T个时刻中某一个时刻的结果就行。
第四行：随机生成一个噪声ϵ，该噪声是用于执行前向过程生成xt=αˉtx0+1−αˉtϵ的。之后，
第五行：我们把xt和t传给神经网络ϵθ(xt,t)，让神经网络预测随机噪声。训练的损失函数是预测噪声和实际噪声之间的均方误差，对此损失函数采用梯度下降即可优化网络。
### 采样
第一行的**x**_t_就是从标准正态分布里随机采样的输入噪声。要生成不同的图像，只需要更换这个噪声。
后面的过程就是扩散模型的反向过程。令时刻从_T_到1，计算这一时刻去噪声操作的均值和方差，并采样出**x**_t_−1。

## Langevin dynamics
在马尔可夫过程中，随机梯度朗温动力学可以通过梯度下降算法从概率密度中产生样本。([Welling & Teh 2011](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf))
使用分数匹配估算的数据分布梯度。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719888917875-2d10d461-2fc0-4a1d-a02c-c9d67ffa1f6c.png#averageHue=%23f5f5f5&clientId=uaf08bd1c-c9f5-4&from=paste&id=o0HfM&originHeight=105&originWidth=698&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=12034&status=done&style=none&taskId=u8ee407a2-56b0-4d19-a610-e7b39b4647a&title=)
![langevin.gif](https://cdn.nlark.com/yuque/0/2024/gif/35698476/1720074896745-1fd103ea-ae56-432b-911e-3a33b8c8bb0a.gif#averageHue=%23f5f2f1&clientId=ua08d167b-6acb-4&from=drop&id=ub887d680&originHeight=432&originWidth=432&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=5753565&status=done&style=none&taskId=ued340a43-4e14-4fb8-913f-05c82df7318&title=)

需要一个得分神经网络𝑠𝜃:𝑅𝐷→𝑅𝐷 用来估算 𝑠𝜃(𝑥)≈∇𝑥log⁡𝑞(𝑥)。

## The score function, score-based models, and score matching
为了建立这样一个生成模型，我们首先需要一种表示概率分布的方法。其中一种方法是直接模拟概率密度函数（p.d.f.）或概率质量函数（p.m.f.），我们将p.d.f.定义为
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719902931151-ee535055-f174-4548-89bf-76c4d527ba68.png#averageHue=%23fcfcfb&clientId=ua70beb0e-0884-4&from=paste&height=69&id=u85bb4429&originHeight=104&originWidth=573&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=5589&status=done&style=none&taskId=u29c6e6e7-138d-46f0-878b-9ebcc41c1b5&title=&width=382)
Z为基于θ的归一化因子，$f_θ(x)$称为非规范化概率模型，或基于**能量的**模型
我们可以通过最大化数据的对数似然来训练𝑝𝜃(𝑥)，就是让我们的数据出现的概率最大化。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719902936297-a9753479-fcf2-442c-bdd7-8e0a3e8a9071.png#averageHue=%23fbfbfa&clientId=ua70beb0e-0884-4&from=paste&height=73&id=u5f1aaa4b&originHeight=109&originWidth=588&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=6560&status=done&style=none&taskId=u80622a8f-9243-4cee-ba58-29c5af0166b&title=&width=392)
然而，方程 (2) 要求𝑝𝜃(𝑥) 是一个归一化概率密度函数。这是不可取的，因为计算𝑝𝜃(𝑥)，我们必须评估归一化常数**𝑍𝜃**，对于任何一般的𝑓𝜃(𝑥)来说，这通常是一个难以解决的问题。因此，为了使最大似然训练可行，基于似然法的模型必须限制其模型结构（例如，自回归模型中的因果卷积，归一化流量模型中的可反转网络），以使𝑍𝜃具有可操作性，或者对归一化常数进行近似（例如，VAE 中的变异推理，或对比发散中使用的 MCMC 采样），这可能会带来高昂的计算成本。
（有一个能量分布，对其进行归一化可以求得其概率P。但计算归一化因子𝑍𝜃很困难，我们要想办法绕过它）

### score function
𝑝(𝑥) 的得分函数定义为**∇𝑥log𝑝(𝑥)**
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719977292377-6106c5bf-a1ca-407c-b33e-df18ab55b3fc.png#averageHue=%23f6f6f3&clientId=u603e3df7-f4cc-4&from=paste&height=468&id=uf89b2ed0&originHeight=935&originWidth=959&originalType=url&ratio=1.5&rotation=0&showTitle=false&size=982418&status=done&style=none&taskId=u69caff09-542e-45ee-94c8-1cf53c1527e&title=&width=480)
### score-based models
通过建立**得分函数模型**而不是密度函数模型，我们可以避免难以解决的归一化常数问题。得分函数的模型称为基于得分的模型，我们将其表示为 **𝑠𝜃(𝑥)**。
score-based models学习目标是：𝑠𝜃(𝑥)≈∇𝑥log𝑝(𝑥)。
例如，我们可以很容易地用公式（1）中定义的基于能量的模型对基于分数的模型进行参数化。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719904768231-5dac5425-213f-4408-9e8f-8c47a3d00abd.png#averageHue=%23f9f8f7&clientId=ua70beb0e-0884-4&from=paste&height=73&id=u94060e98&originHeight=109&originWidth=844&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=9450&status=done&style=none&taskId=u742c6d69-b817-4c08-b92a-f7b1a3d284b&title=&width=562.6666666666666)
![](https://cdn.nlark.com/yuque/0/2024/gif/35698476/1719905060578-62101c3b-5b25-4475-ba26-7d3144ad1e72.gif#averageHue=%23fbfbfb&clientId=ua70beb0e-0884-4&from=paste&id=u83a16ba3&originHeight=255&originWidth=360&originalType=url&ratio=1.5&rotation=0&showTitle=false&status=done&style=none&taskId=u30e0a50b-2df9-4910-b91a-b92ab85d933&title=)![score.gif](https://cdn.nlark.com/yuque/0/2024/gif/35698476/1719905176571-03624cc2-e829-494e-a988-ff212a619255.gif#averageHue=%23fcfcfc&clientId=ua70beb0e-0884-4&from=drop&id=u760b552b&originHeight=241&originWidth=360&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=345870&status=done&style=none&taskId=u15e124d4-396d-4b7a-8c14-c26a8beaf42&title=)
pdf需要归一化保证曲线下面积为1，score不用
训练模型最小的目标：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719905341920-6ba7eb01-79c7-4bde-99f1-029941c9f121.png#averageHue=%23faf8f7&clientId=ua70beb0e-0884-4&from=paste&height=45&id=uc7fb067b&originHeight=67&originWidth=634&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=6808&status=done&style=none&taskId=ue789488f-1b69-4c62-a912-43e314d1475&title=&width=422.6666666666667)
### score matching
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719905341920-6ba7eb01-79c7-4bde-99f1-029941c9f121.png#averageHue=%23faf8f7&clientId=ua70beb0e-0884-4&from=paste&height=45&id=EHizF&originHeight=67&originWidth=634&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=6808&status=done&style=none&taskId=ue789488f-1b69-4c62-a912-43e314d1475&title=&width=422.6666666666667)
然而，Fisher divergence是不可直接计算的，因为它需要获取未知的数据得分∇𝑥log𝑝(𝑥)。需要通过一系列称为**score matching**的方法，可以在**不知道真实数据分数**的情况下最小化Fisher divergence。
**score matching**可以直接在现有的数据集上进行估计，并通过随机梯度下降法进行优化，类似于训练基于似然模型（已知归一化常数）的对数似然目标。
Fisher divergence本身并不要求𝑠𝜃(𝑥) 是任何归一化分布的实际得分函数，它只需**比较真实数据**得分与基于得分的模型之间的 ℓ2 距离，而无需对𝑠𝜃(𝑥) 的形式做出额外的假设。事实上，对基于分数的模型的唯一要求就是它应该是一个输入和输出维度相同的向量值函数，这在实践中很容易满足。 
## 
## score-based models和stochastic differential equations（SDG）
### 常微分方程 (ODE) 和 随机微分方程 (SDE)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1720061960762-8a0fa8e6-e0b1-4828-b327-3f49e2ba1843.png#averageHue=%23fcfbfa&clientId=ua08d167b-6acb-4&from=paste&id=ua3d9c901&originHeight=378&originWidth=862&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=31561&status=done&style=none&taskId=uc805bf37-a545-48f1-ba21-04e58bc9ec7&title=)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1720063221536-07b99cd7-0db7-4f14-89e6-1e4844ac3fcf.png#averageHue=%23fbfafa&clientId=ua08d167b-6acb-4&from=paste&id=ucbf7f030&originHeight=301&originWidth=989&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=34373&status=done&style=none&taskId=u5f57320f-f759-4ddb-b500-bdb3147eaf5&title=)
### 前向过程
随机微分方程：常微分方程加上一个白噪音项
许多随机过程（尤其是扩散过程）都是随机微分方程（SDE）的解。一般来说，SDE 具有以下形式：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719907593717-08a73e6f-e78e-4ac4-8b2b-e2f12e239128.png#averageHue=%23f9f7f5&clientId=ua70beb0e-0884-4&from=paste&height=125&id=u4621a7ed&originHeight=188&originWidth=915&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=30445&status=done&style=none&taskId=u3059242b-ad9e-4f25-b0c6-65faa8bf2bd&title=&width=610)
dx是$x_t-x_{t-1}$的变化量

𝑝0(𝑥)=𝑝(𝑥) 是数据分布，因为在 𝑡=0 时没有对数据进行扰动。用随机过程扰动𝑝(𝑥)足够长的时间𝑇后，𝑝𝑇(𝑥)会变得接近一个可控的噪声分布𝜋(𝑥)，称为**prior distribution**。我们注意到，𝑝𝑇(𝑥) 类似于有限噪声尺度情况下的𝑝𝜎𝐿(𝑥)，这相当于对数据施加最大的噪声扰动𝜎𝐿。

### 逆转样本生成的SDE
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719926788471-d9b7a616-09e8-4b54-8168-54a5bbb74cd8.png#averageHue=%23f8f7f6&clientId=u60966e75-0bd8-4&from=paste&height=51&id=ub19394cb&originHeight=77&originWidth=749&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=8135&status=done&style=none&taskId=ub44319d5-5170-49bc-9ffa-5fcbd6be2a8&title=&width=499.3333333333333)
这里的 d𝑡 代表负的无穷小时间步长，因为 SDE需要逆向求解（从 𝑡=𝑇 到 𝑡=0）。为了计算反向 SDE，我们需要估计 ∇𝑥log𝑝𝑡(𝑥)，这正是 𝑝𝑡(𝑥)的得分函数。
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719927043636-e9d167f3-c7a2-4896-8b6a-8c0dbddbe493.png#averageHue=%23e6e4e1&clientId=u60966e75-0bd8-4&from=paste&id=u50f233b7&originHeight=1030&originWidth=2383&originalType=url&ratio=1.5&rotation=0&showTitle=false&size=2170205&status=done&style=none&taskId=u7bcacfec-d356-490f-9e45-eb29af5089c&title=)
### 用基于分数的模型和分数匹配估算反向 SDE
求解反向 SDE 要求我们知道最终的分布𝑝𝑇(𝑥)和得分函数∇𝑥log𝑝𝑡(𝑥)。前者接近于**prior distribution **𝜋(𝑥)。为了估算∇𝑥log𝑝𝑡(𝑥)，我们训练了一个基于时间依赖性分数的模型𝑠𝜃(𝑥,𝑡)，使得𝑠𝜃(𝑥,𝑡)≈∇𝑥log𝑝𝑡(𝑥)。这类似于用于有限噪声尺度的基于噪声条件的分数模型𝑠𝜃(𝑥,𝑖)，经过训练后，𝑠𝜃(𝑥,𝑖)≈∇𝑥log𝑝𝜎𝑖(𝑥)。
我们对 𝑠𝜃(𝑥,𝑡)的训练目标是Fisher divergences的连续加权组合，其值为：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719927645539-e1a67b1a-73fc-4985-9e69-11ad727eee64.png#averageHue=%23f9f7f6&clientId=u60966e75-0bd8-4&from=paste&height=47&id=ua8d5ab17&originHeight=71&originWidth=750&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=9776&status=done&style=none&taskId=u16a63800-e3de-4441-9bb7-fdca930232a&title=&width=500)
其中，𝑈(0,𝑇) 表示时间区间 [0,𝑇] 上的均匀分布，𝜆:𝑅→𝑅>0 是正加权函数。通常情况下，我们使用𝜆(𝑡)∝1/𝐸[‖∇𝑥(𝑡)log𝑝(𝑥(𝑡)∣𝑥(0))‖22]来平衡不同时间内不同分数匹配损失的大小。
与之前一样，我们的Fisher divergences可以通过score matching方法进行有效优化。一旦我们基于分数的模型 𝑠𝜃(𝑥,𝑡)训练到最优，我们就可以将其插入(10)中的反向 SDE 表达式，从而得到估计的反向 SDE。

我们可以从 𝑥(𝑇)∼𝜋 开始，求解上述反向 SDE，得到样本 𝑥(0)。我们把这样得到的𝑥(0) 分布称为𝑝𝜃。当基于分数的模型𝑠𝜃(𝑥,𝑡)训练有素时，我们有𝑝𝜃≈𝑝0，在这种情况下，𝑥(0) 是数据分布𝑝0 的近似样本。

## 
## Classifier Guidance
比如：引入一些提示词、结构中就引入一些能量监督的形式

![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719994783872-b67c528c-7c4b-4486-9029-f35eaa3af43b.png#averageHue=%23fcfaf8&clientId=u5e897f93-3a19-4&from=paste&id=u3fef6c86&originHeight=45&originWidth=416&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=6960&status=done&style=none&taskId=u4c7cff86-4989-48d9-903f-0d1b611d773&title=)
利用贝叶斯公式，对∇xlogp (x∣y)进行处理：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719994853451-478d49f3-20bd-4372-83b0-3b424306744f.png#averageHue=%23fcfbfa&clientId=u5e897f93-3a19-4&from=paste&id=u44e0236b&originHeight=146&originWidth=490&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=18855&status=done&style=none&taskId=uedc77df9-9770-4565-89d6-7e57c4e458f&title=)
加入landa进行控制：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1719994935911-134ce1a8-5286-4878-87b4-c1d64ebb3c14.png#averageHue=%23fcfbfa&clientId=u5e897f93-3a19-4&from=paste&id=u6d1b26a8&originHeight=61&originWidth=451&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=6040&status=done&style=none&taskId=ue5394ad3-55ef-4f96-ba95-952716045b2&title=)

这种方法的一个缺点就是，需要额外学习一个分类器 pt(y|x)


## Classifier-Free Guidance
......

最大似然估计
![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1716823665400-bdb56655-a35a-4861-86b9-cb6c7f3a168d.png#averageHue=%23f7f6f6&clientId=u8c027539-ab4a-4&from=paste&height=826&id=u983b03b6&originHeight=1239&originWidth=2270&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=621362&status=done&style=none&taskId=u6d93f8b4-589e-4243-b7b8-acd741e5524&title=&width=1513.3333333333333)
把采样到的x到xm都求一下对应的$P_θ$，然后全部乘起来，找出最大的θ




ddpm分为前向过程以及反向过程：

- 前向过程：逐步将加入随机高斯噪声
- 反向过程：逐步消除高斯噪声
- 优化目标：加入的噪声与消除的噪声越近越好

sde：

- 前向过程：以SDE（随机微分方程）定义这个过程，对数据进行扰动。![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1720267994009-ca337ed7-8ed4-4545-94ed-444d05f34d64.png#averageHue=%23efefef&clientId=u8e5d7419-ffdc-4&from=paste&height=35&id=tx9fO&originHeight=185&originWidth=810&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=34714&status=done&style=none&taskId=uc91115c8-722e-4d38-9641-2061c4651eb&title=&width=154)f(x,t)是已经确定的函数，状态函数。同样也是加入随机高斯噪声，但这个噪声符合维纳过程，称为布朗运动。
- 反向过程：![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1720268191595-d63babcd-174d-4e78-8f56-76a7a4aa26fb.png#averageHue=%23f0f0f0&clientId=u8e5d7419-ffdc-4&from=paste&height=30&id=v39J4&originHeight=212&originWidth=1534&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=94813&status=done&style=none&taskId=uac36f83c-c39b-4f30-a062-1154ed3ac6c&title=&width=215.3333740234375)。方框里的称为得分，即概率对数的梯度。
- 如何得到∇xlogpt(x)：
   - ![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1720272610883-8a3f6dde-b08a-4e6b-ac8a-a56b12e6a83d.png#averageHue=%23ffffff&clientId=u8e5d7419-ffdc-4&from=paste&height=38&id=YS5N6&originHeight=104&originWidth=785&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=9791&status=done&style=none&taskId=ua98c63ee-7904-44c2-ad05-415cee0dbac&title=&width=284.3333740234375)通过前向过程可以估算
   - 此上式实际上是能够直接估算的，然而它涉及到对全体训练样本x0的平均，一来计算量大，二来泛化能力也不够好。因此，我们希望用神经网络学一个函数𝑠𝜃(𝑥𝑡,𝑡)，使得它能够直接计算
   - 通过最小化![image.png](https://cdn.nlark.com/yuque/0/2024/png/35698476/1720272936989-1bb60217-9d8f-4c0c-8dd9-c5edf25d13c8.png#averageHue=%23ffffff&clientId=u8e5d7419-ffdc-4&from=paste&height=33&id=jptMa&originHeight=80&originWidth=766&originalType=binary&ratio=1.5&rotation=0&showTitle=false&size=11400&status=done&style=none&taskId=uff1bb4f3-c392-4fc4-9e91-47c00ed90b8&title=&width=319.66668701171875)来训练神经网络𝑠𝜃(𝑥𝑡,𝑡)

