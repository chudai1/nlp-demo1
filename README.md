# NLP 文本分类任务 (基于 BERT)

本项目是一个基于 PyTorch 和 Hugging Face Transformers 构建的 15 分类中文文本分类模型。

## 💡 项目特点
- **完全解耦的架构**：将数据、模型、训练、配置分离，高度模块化。
- **手写底层逻辑**：没有使用 `BertForSequenceClassification` 和 `Trainer` 等高层封装，而是通过继承 `nn.Module` 手写 BERT + Linear 分类头及 CrossEntropyLoss，更好地控制底层逻辑。
- **多种子实验**：采用 3 个不同的随机种子（42, 66, 99）进行独立训练，避免偶然性。
- **早停机制**：根据验证集表现（Dev Acc）自动保存最优权重，并触发 Early Stopping 防止过拟合。
- 
🚀 快速开始
1. 环境依赖

pip install torch transformers swanlab tqdm numpy

2. 运行训练

python main.py

📊 实验结果
在测试集上的最终表现（平均 3 个种子）：

测试集准确率: 83.65% ± 0.58%

训练过程中的 Loss 曲线与指标已通过 SwanLab 进行记录与可视化。

## 📁 目录结构
```text
├── config.py       # 全局配置与超参数
├── data.py         # 数据清洗与 Dataset 封装
├── model.py        # 手写 TextClassifier 模型类 (继承 nn.Module)
├── trainer.py      # 核心训练循环与早停逻辑
├── utils.py        # 模型评估函数
└── main.py         # 程序入口

