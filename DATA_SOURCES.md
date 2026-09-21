# 实验资料中的数据来源

这些资料是公开数据集的研究衍生物，不是自有硬件采集数据；资料包不包含官方原始数据全集。请保留本页与各数据来源的许可和署名。

## PPG-DaLiA

来源：Reiss, A., Indlekofer, I., & Schmidt, P. (2019), PPG-DaLiA，UCI Machine Learning Repository，[DOI 10.24432/C53890](https://doi.org/10.24432/C53890)。
[官方数据页面](https://archive.ics.uci.edu/dataset/495/ppg%20dalia) 标示 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

第一阶段窗口特征、心率标签统计和第三阶段 E4/ACC 样例来自该数据集。V3 与 ACC 一致性配对数据经过筛选、切窗、标准化及人工污染；它们不是原始文件，也不是真实运动的同步污染—干净真值。

SMoLK 固定回放的 input_normalized.npy 来自 [作者固定提交](https://github.com/SullyChen/SMoLK/tree/4aa63d72d5fedb80bd5ba6a905dcc75f1cafd12a) 的 DaLiA 测试第 0 行，经逐段标准化。expected_probability、expected_savgol151_probability 和 expected_mask 是本项目模型计算的输出；absorbed_model.npz 是训练、剪枝后的模型参数。作者的完整 DaLiA/WESAD/TROIKA 数组不随资料包提供。

## BIDMC

来源：Marco Pimentel、Alistair Johnson、Peter Charlton、David Clifton，BIDMC PPG and Respiration Dataset，PhysioNet，version 1.0.0，2018，[DOI 10.13026/C2208R](https://doi.org/10.13026/C2208R)。
[官方页面](https://physionet.org/content/bidmc/1.0.0/) 标示 [Open Data Commons Attribution License v1.0](https://physionet.org/content/bidmc/view-license/1.0.0/)。

V2 配对数据包含从 BIDMC 提取并重采样、带限和标准化的 PPG，以及由 PPG-DaLiA ACC 等成分构造的人工污染。保留上述两类来源说明；不能把 V2 数组整体标成 SMoLK 的 MIT 代码数据。

原数据页面要求引用 Pimentel 等的研究：Towards a Robust Estimation of Respiratory Rate from Pulse Oximeters，[DOI 10.1109/TBME.2016.2613124](https://doi.org/10.1109/TBME.2016.2613124)。

## 本项目边界

资料包中的 CSV、图、固定样例和九个配对 NPZ 保留既有研究字节；本次没有重新划分数据、生成新污染或修改历史成绩。对真实逐搏恢复的限制仍适用。

数据来源许可与本项目新增代码的许可分别处理。本项目新增内容的公开范围尚未确定；存档仓库保持私有，不把上游许可解释为对工作成果公开发布的授权。
