# 第三方来源与许可范围

[返回首页](README.md)

## SMoLK

模型依据 Sully Chen 等人的 [SMoLK 论文](https://arxiv.org/html/2307.05385v4) 与 [固定作者 notebook](https://github.com/SullyChen/SMoLK/blob/4aa63d72d5fedb80bd5ba6a905dcc75f1cafd12a/PPG%20Segmentation.ipynb)，代码提交为 4aa63d72d5fedb80bd5ba6a905dcc75f1cafd12a。

原作者 MIT 许可完整保留在 [LICENSE-SMoLK](research/02_论文复现与验证/01_SMoLK论文复现/LICENSE-SMoLK)，其中版权为 Copyright (c) 2024 Sully Chen。模型设计归原作者，本项目的实现核查、基线、对照与重建探索按各案例说明区分。

该 MIT 文本不自动覆盖本项目新增内容，也不代表 PPG 数据、论文全文或图表都使用同一许可。本仓库当前保持私有，新增内容的公开许可另行确定。

## 数据

- [PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg%2Bdalia)：原始 BVP、ACC、活动及 ECG 来源标签。
- [BIDMC](https://physionet.org/content/bidmc/1.0.0/)：用于 V2 配对研究的带限参考来源。
- SMoLK 发布的 DaLiA/WESAD/TROIKA 处理后数组由复现脚本从固定作者提交读取。

本次资料包另纳入本项目的模型权重、统计结果、固定样例和九个配对数组；不提供官方原始全集与作者完整处理数组。数据作者、许可及具体变换见 [DATA_SOURCES.md](DATA_SOURCES.md)，同一说明也放入每个 ZIP。使用者应保留并遵循各数据来源的说明和许可，不从模型代码许可推断数据许可。
