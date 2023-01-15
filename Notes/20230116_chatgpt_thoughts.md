# chatgpt反思

https://mp.weixin.qq.com/s/eMrv15yOO0oYQ-o-wiuSyw

### 潮流之巅：NLP研究范式的转换

* ##### 范式转换1.0

  * LSTM -> seq2seq+attn -> Bert / GPT
  * 主要在思考怎么增加参数量

* ##### 1.0影响

  * 中间任务消亡
    * NER，parser本来就是机器翻译产生的中间任务
    * 语言特征都在参数里
  * 不同研究方向技术路线核心
    * 分类 / 生成
    * pretrain + finetune (Bert)
    * Zero / few shot prompt (GPT)

* ##### 范式转换2.0

  * PLM -> AGI 通用人工智能
  * 过渡期：自回归+prompting
    * 为什么？
    * T5统一了NLU和NLG，NLU不兼容NLG但是NLG兼容NLU
    * 我们最终追求的是zero / few shot，LLM适配人而不是人适配LLM
    * 在追求zero shot的时候不尽如人意，退而求其次few shot

* ##### 影响

  * “LLM适配人”的接口
    * 注入“人类偏好”的知识
  * 许多领域开始消亡
    * LLM超过人类的（去看数据集）
    * few shot超过finetune的
  * 其他很多领域开始纳入LLM
    * AGI！代码，音乐，多模态...

### 学习者：从无尽数据到海量知识

* ##### 求知之路：LLM学到了什么知识

  * 语言类
  * commonsense：LLM是隐式的知识图谱

* ##### 记忆之地：LLM如何存取知识

  * MHA：“知识之间的联系”
  * FFN：ffn是key-value存储器

* ##### 知识涂改液：如何修正LLM里存储的知识

  * 训练数据源头
  * finetune，新的记忆，忘记旧的
  * 修改参数：具体定位ffn位置

* ##### LLM越来越大会发生什么

  * GPT3 175B，LaMbDa 137B，PaLM 540B
  * scaling law
    * 数据量，参数，训练时长共同决定，增加要同样比例增加
    * 怎么分配资源？比如10xGPU就要3x数据，3x参数量
  * 随着规模越来越大，任务表现更好
    * 多为知识密集型任务
  * 涌现能力
    * 多为多阶段，多步骤任务
    * 有可能中间所有步骤都对了才行
  * U型变化

### 人机接口：从in context learning到instruct

* ##### ICL (in context learning = few shot learning)

  * 为什么work？
    * 答案对不对并不重要
    * 重要的是给出的(x,y)中x和y的分布

* ##### Instruct (zero shot learning)

  * 如何增强能力？
    * 增加任务数量
    * 增加大小
    * CoT
    * 增加任务多样性

* ##### ICL vs Instruct

  * 有价值的研究：few shot example -> instruct
  * LLMs are human level prompt engineers

### 智慧之光：如何增强LLM的推理能力

* ##### prompt (google)

  * 辅助推理prompt：zero shot CoT
    * think step by step
    * 可能激发了模型某种回忆，激发模型的能力
  * few shot CoT
    * 给出中间的推理过程
  * self consistency
    * 输出多个结果，majority vote
  * 分治算法
    * X->Y，先问模型如果要得到Y，我们需要的上一步是什么（prompt）

* ##### 引入程序代码 (openai)

* ##### 更可行的思路：分治算法

### 未来之路：LLM研究趋势和重点方向

* 探索LLM上限
* 增强推理能力
  * 为什么代码预训练有效？
* LLM更多领域，AGI
* 更容易的交互接口
* 更综合的评测数据集
* 更高质量的数据集
* Transformer稀疏化
  * 更快训练
