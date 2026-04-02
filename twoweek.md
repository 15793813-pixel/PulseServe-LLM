二、第 1 周：先把“程序骨架”和“推理主流程”打通
1. Python 项目结构

这一块你不是去学“Python 语法”，而是学：

一个稍微像样的工程，应该怎么组织。

1）必须学的内容
a. 目录结构怎么分层

至少要理解这些文件/目录的作用：

main.py
src/
tests/
configs/
requirements.txt 或 pyproject.toml
README.md
__init__.py

你要知道：

main.py 通常是入口
src/ 放核心代码
tests/ 放测试
configs/ 放配置
README.md 说明怎么运行
requirements.txt/pyproject.toml 管依赖
__init__.py 让目录变成 package
b. import 机制

这块很重要，很多人项目一大就死在这。

你要学会：

相对导入、绝对导入
from xxx import yyy
Python 怎么找模块
为什么会报 ModuleNotFoundError
工作目录和包路径的关系

你至少要能解释：

为什么有时候直接 python file.py 能跑
为什么换个目录运行就 import 不到了
c. 面向模块写代码

你要练习把功能拆成小模块：

比如一个最小推理项目可以拆成：

tokenizer.py
model.py
sampler.py
generate.py
main.py

你要形成习惯：

一个文件只负责一类事。

d. 配置与参数管理

你至少要会：

用 argparse 读命令行参数
用 json/yaml 保存配置
区分“代码逻辑”和“超参数配置”

例如：

模型路径
最大生成长度
temperature
top-k
device

这些别写死在代码里。

e. 日志、异常、调试

你要会：

print 调试
logging 基础
try/except
看 traceback

因为以后做推理服务，报错特别常见：

shape 不匹配
device 不一致
dtype 不一致
tokenizer 输出不符合预期
2）学到什么程度算够

前两周不需要你会搞特别标准的大型工程架构。
你只要做到：

能自己搭一个 5~10 个文件的小项目
不乱 import
能把功能拆开
能写一个简单配置系统
能运行入口脚本

就够了。

3）这一块为什么重要

因为你后面不是写 notebook 玩具，而是要做一个像样的推理项目。
如果项目结构乱，你后面加：

KV cache
batch 推理
streaming
benchmark
tests

时会直接崩。

2. PyTorch 推理

这部分是第 1 周最核心的。

你现在先不要从“训练”入手，先专门学推理。

1）必须学的内容
a. Tensor 基础

先把这些打牢：

创建 tensor
shape / ndim
view, reshape
unsqueeze, squeeze
transpose, permute
索引、切片
cat, stack
广播机制

你要看到一个张量时，脑子里能立刻反应：

它是什么维度
每一维代表什么
变形以后语义还对不对

例如 LLM 常见：

[batch, seq_len]
[batch, seq_len, hidden]
[batch, heads, seq_len, head_dim]
b. nn.Module 与 forward

你要会：

自定义一个 nn.Module
写 __init__
写 forward
知道参数在哪里
知道前向传播是怎么跑的

哪怕先写一个很小的模型：

Embedding
Linear
LayerNorm
简单双层 MLP

先理解“输入 tensor 经过一层层模块变成输出”。

c. 推理模式

这个特别重要。

你必须学：

model.eval()
torch.no_grad()
torch.inference_mode()

并理解：

为什么推理不需要梯度
为什么这样能省显存、省开销
eval() 和 no_grad() 不是一回事

你至少要分清：

eval()：切换 dropout / batchnorm 行为
no_grad()：不追踪梯度
inference_mode()：更激进的推理优化
d. device 与 dtype

必须掌握：

CPU / CUDA device
.to(device)
.cuda()
.cpu()
float32, float16, bfloat16, int64

你要能看懂这些问题：

为什么 embedding 输入通常是 int64
为什么模型权重常用 fp16/bf16
为什么 logits 往往还是 float 类型
为什么 device 不一致会报错
e. 前向推理流程

你要能完整理解这条链：

input ids → embedding → 模型前向 → logits → 取最后一个位置 → sampling → next token

这就是 LLM 生成的主干。

哪怕你现在不看 Transformer 全细节，也要先把这个“主循环”理解清楚。

2）进阶但前两周也要碰一下的内容
a. batch 推理

理解：

为什么多个样本可以一起推
batch 维度的意义
padding 的作用
attention mask 的必要性
b. 显存与速度基础认知

你要初步知道：

batch 越大，显存越高
seq_len 越长，显存越高
dtype 越低，显存占用越小
不关梯度会浪费显存
c. 常见层的输入输出 shape

先熟悉：

Embedding
Linear
LayerNorm
Softmax
3）学到什么程度算够

第 1 周结束，你应该能自己写出这种级别的代码：

一个 toy tokenizer 输出 ids
一个 toy model 接收 ids 输出 logits
用 torch.inference_mode() 做前向
从 logits 里拿最后一个 token 分布
采样出下一个 token

不用训练。
先会“跑通推理”。

三、第 2 周：把 tokenizer / sampling / GPU tensor 串起来
3. tokenizer

这是 LLM 入门里特别容易被忽略，但其实极其关键的部分。

1）你必须理解的核心问题
a. 为什么不能直接按“字”或“词”输入模型

你要明白：

模型只认识数字，不认识文字
tokenizer 的职责是把文本映射成 token id
token 不一定等于字，不一定等于词
b. token、id、vocab 的关系

必须搞清：

token：切出来的符号单元
vocab：词表
token id：token 在词表中的索引

例如：

文本 → token 序列
token 序列 → id 序列
id → embedding 向量
c. 编码和解码

你要会：

encode：文本转 ids
decode：ids 转文本

并理解：

为什么 decode 不一定完全还原原始空格/格式
为什么不同 tokenizer 结果差很多
d. 子词切分思想

你不一定要深挖算法实现，但要知道：

BPE
WordPiece
SentencePiece / Unigram

它们的基本思想都是：

把文本拆成高频、可复用的子词单元。

e. 特殊 token

你要知道这些特殊 token 可能干什么：

BOS
EOS
PAD
UNK

尤其是：

EOS 为什么能表示“该停了”
PAD 为什么对 batch 很重要
2）前两周建议学到什么程度

你要做到：

能用现成 tokenizer 编码/解码
能打印一段文本的 token 切分结果
能知道一个句子切成多少 token
能理解为什么中文、英文的切分不一样
能理解 prompt 最终就是一串 ids
3）你还要额外理解：位置不是 tokenizer 负责的

很多初学者会混。

你要分清：

tokenizer 负责“文字 → token ids”
position encoding / positional embedding 负责“顺序信息”

这是两层事。

4. sampling

这是生成质量非常核心的一部分。

模型输出 logits 后，并不是直接“吐文字”，中间还要经过采样策略。

1）必须学的内容
a. logits 是什么

你要理解：

模型最后输出的是每个词表位置的分数
这个分数还不是概率
需要 softmax 才能变成概率分布
b. greedy decoding

最简单的：

每次都选概率最大的 token

你要知道优缺点：

稳定
便宜
但容易单调、重复
c. temperature

你必须理解它的直觉：

temperature 低：分布更尖锐，更保守
temperature 高：分布更平，更随机

本质上是在调 logits 的“尖锐程度”。

d. top-k sampling

你要理解：

只保留概率最高的前 k 个 token
其余砍掉
再重新归一化采样

作用：

防止从太离谱的长尾 token 里乱抽
e. top-p sampling

你要理解：

按概率从高到低累加
保留累计概率达到 p 的最小集合
再从里面采样

它比 top-k 更“自适应”。

f. repetition penalty / 基础防重复思想

前两周至少知道概念：

为什么模型会重复
怎么通过惩罚已生成 token 来缓解
g. 停止条件

必须知道：

生成到 EOS 停
达到 max_new_tokens 停
某些特定 stop token / stop string 停
2）学到什么程度算够

你要至少能自己实现：

greedy
temperature
top-k
top-p

哪怕代码比较朴素也行。

你要真正明白：

生成质量 = 模型能力 + sampling 策略。

同一个模型，sampling 不同，输出差很多。

四、GPU tensor 基础

你这两周不需要学 CUDA 编程，但必须学 GPU 张量基础。

因为你做推理服务，不懂这个会寸步难行。

1）必须学的内容
a. CPU tensor 和 GPU tensor 的区别

你要理解：

tensor 数据实际存在哪里
CPU tensor 在主存
GPU tensor 在显存
运算在哪个设备上，数据就最好也在哪个设备上
b. 数据搬运

必须掌握：

x.to("cuda")
x.to(device)
x.cpu()
x.cuda()

并理解：

CPU?GPU 拷贝是有成本的
不要来回搬运
推理主循环里频繁搬运会很慢
c. 显存是什么在占

你要知道显存主要可能被谁占：

模型参数
中间激活
KV cache
batch 数据
临时 tensor

虽然 KV cache 你后面再深学，但现在要先知道它存在。

d. contiguous / view / reshape 的直觉

至少理解：

有些 tensor 内存布局不是连续的
view 有时要求 contiguous
reshape 相对更宽松
permute/transpose 后常出现布局变化

这在后面做性能优化和 debug 很常见。

e. dtype 对显存和速度的影响

要知道：

fp32 更稳但更占空间
fp16/bf16 更省显存、更快
int tensor 常用于 token ids
不同 dtype 混合时可能出问题
f. 常见报错

你要熟悉这些报错背后的本质：

Expected all tensors to be on the same device
expected scalar type Half but found Float
CUDA out of memory
shape mismatch

别只会复制报错，得知道问题出在哪。

2）建议你顺手学一点的内容
a. GPU 异步执行

先建立直觉：

很多 CUDA 操作是异步的
有时候你看到代码跑到下一行了，GPU 还没真干完
计时时常需要同步

这对你以后 benchmark 很重要。

b. 基本性能意识

你先有这个概念就行：

小 tensor 频繁操作不一定快
Python 循环会拖后腿
能张量化就张量化
频繁 .item()、.cpu().numpy() 可能拖慢速度
3）前两周学到什么程度算够

你不需要会写 CUDA kernel。
但你必须做到：

能把模型放到 GPU 上
能把输入放到 GPU 上
能知道输出为什么还在 GPU 上
能处理 device/dtype 报错
能初步理解显存占用来源
五、你这两周每天到底怎么学

我给你一个更细的安排。

第 1 周：先把程序和推理主链路打通
Day 1

学 Python 项目结构：

目录结构
包与模块
__init__.py
import 机制
main.py 入口

目标：

自己搭一个小项目骨架
Day 2

学配置和工程习惯：

argparse
配置文件
logging
异常处理

目标：

能从命令行读参数运行脚本
Day 3

学 PyTorch tensor 基础：

创建 tensor
shape 操作
索引切片
reshape / permute / cat / stack
广播

目标：

看到 shape 不发怵
Day 4

学 nn.Module 和简单前向：

自定义模块
Embedding
Linear
LayerNorm
forward

目标：

写一个 toy model
Day 5

学推理模式：

eval
no_grad
inference_mode
device / dtype

目标：

能在 CPU/GPU 上跑一个前向
Day 6

学完整推理最小链路：

input ids
embedding
forward
logits
取最后一位 logits

目标：

手写一个“预测下一个 token”的 demo
Day 7

复盘 + 小练习：

重写一遍最小推理项目
不看资料自己搭结构
排查一次 shape/device 错误
第 2 周：补 tokenizer / sampling / GPU 直觉
Day 8

学 tokenizer 基础：

token / vocab / id
encode / decode
特殊 token

目标：

打印一句话的 tokenization 结果
Day 9

学子词切分思想：

BPE
WordPiece
SentencePiece 的直觉
中文英文切分差异

目标：

真正理解“token 不等于字”
Day 10

学 logits 和 softmax：

logits 是什么
softmax 变概率
为什么只看最后一个位置

目标：

把模型输出转成概率分布
Day 11

学 greedy、temperature

目标：

自己实现最基础采样器
Day 12

学 top-k、top-p、停止条件

目标：

实现一个小型 sampler.py
Day 13

学 GPU tensor 基础：

CPU/GPU 搬运
显存概念
dtype
contiguous/view/reshape 直觉

目标：

在 GPU 上跑通生成一步
Day 14

做一个综合 mini project：

输入字符串 → tokenizer → ids → toy model → logits → sampling → next token / 多步生成

这就是你这两周的验收项目。

六、每个方向你要重点关注的“知识关键词”

你后面自己找资料时，就按这些关键词搜。

1. Python 项目结构关键词
package / module
__init__.py
absolute import / relative import
argparse
logging
requirements.txt
pyproject.toml
entrypoint
config management
2. PyTorch 推理关键词
tensor shape
nn.Module
forward
model.eval()
torch.no_grad()
torch.inference_mode()
device / dtype
embedding
logits
3. tokenizer 关键词
token
vocab
token id
encode / decode
BPE
WordPiece
SentencePiece
BOS / EOS / PAD / UNK
4. sampling 关键词
logits
softmax
greedy decoding
temperature
top-k
top-p / nucleus sampling
repetition penalty
stop condition
5. GPU tensor 基础关键词
CPU vs GPU tensor
.to(device)
CUDA memory
fp32 / fp16 / bf16
contiguous
view vs reshape
async execution
out of memory
七、这两周你不该过早陷进去的东西

前两周先别太深挖这些：

CUDA kernel 编程
Triton kernel
FlashAttention 实现细节
分布式推理
Tensor Parallel / Pipeline Parallel
quantization 底层实现
vLLM 源码细节
Transformer 全数学推导

不是说不重要，
而是你现在先要把：

“文本 → token → tensor → forward → logits → sampling → 输出”

这条主链真正吃透。