# 研究源码导航

[返回首页](../README.md) · [运行与数据准备](../RUNNING.md)

本目录按原工程结构保存 35 个 Python 脚本，逐字节复制，不拆分函数、不改变算法与实验规则。校验值见 [SOURCE_SHA256SUMS](../SOURCE_SHA256SUMS)。

先看心率基线，再看 SMoLK，最后看 ResidualPPG；完整 SQI＋ACC、时间映射、V3 和一致性回放的工件依赖见运行说明。现存配置、权重和结果见 [资料包](../ARTIFACTS.md)，缺失历史输出与旧源码差异见 [复现核查](../REPRODUCTION_AUDIT.md)。

## 第一阶段：自主基线

| 文件 | 职责 |
|---|---|
| [analyze_all_subjects_ppg_sqi.py](01_自主基线研究/01_代码/analyze_all_subjects_ppg_sqi.py) | 全受试者 PPG 特征表 |
| [analyze_candidate_feature_separation_v0.py](01_自主基线研究/01_代码/analyze_candidate_feature_separation_v0.py) | 候选特征区分探索 |
| [analyze_candidate_peak_recoverability_v0.py](01_自主基线研究/01_代码/analyze_candidate_peak_recoverability_v0.py) | 候选峰恢复探索 |
| [analyze_s1_motion_artifacts.py](01_自主基线研究/01_代码/analyze_s1_motion_artifacts.py) | S1 运动伪影分析 |
| [analyze_s1_ppg_sqi.py](01_自主基线研究/01_代码/analyze_s1_ppg_sqi.py) | S1 PPG 质量特征 |
| [build_acc_motion_features_v0.py](01_自主基线研究/01_代码/build_acc_motion_features_v0.py) | ACC 运动特征 |
| [evaluate_acc_aware_sqi_v1.py](01_自主基线研究/01_代码/evaluate_acc_aware_sqi_v1.py) | SQI＋ACC 规则设计 |
| [evaluate_ppg_only_sqi_v0.py](01_自主基线研究/01_代码/evaluate_ppg_only_sqi_v0.py) | PPG SQI 设计与评价 |
| [evaluate_temporal_consistency_v0.py](01_自主基线研究/01_代码/evaluate_temporal_consistency_v0.py) | 时序一致性探索 |
| [inspect_s1_ppg_dalia.py](01_自主基线研究/01_代码/inspect_s1_ppg_dalia.py) | S1 数据结构检查 |
| [plot_s1_alignment.py](01_自主基线研究/01_代码/plot_s1_alignment.py) | PPG/ECG HR 时间对齐图 |
| [ppg_hr_baseline.py](01_自主基线研究/01_代码/ppg_hr_baseline.py) | 频谱心率基线 |
| [validate_acc_aware_sqi_v1.py](01_自主基线研究/01_代码/validate_acc_aware_sqi_v1.py) | 冻结规则验证 |

## 第二阶段：SMoLK 复现

| 文件 | 职责 |
|---|---|
| [deployment_numpy_runtime.py](02_论文复现与验证/01_SMoLK论文复现/deployment_numpy_runtime.py) | NumPy 框架外推理 |
| [recover_dalia_mapping.py](02_论文复现与验证/01_SMoLK论文复现/recover_dalia_mapping.py) | 处理后片段的原始时间映射 |
| [run_deployment_replay.py](02_论文复现与验证/01_SMoLK论文复现/run_deployment_replay.py) | 电脑端导出与固定回放 |
| [run_full.py](02_论文复现与验证/01_SMoLK论文复现/run_full.py) | 完整单种子训练与三数据集评价 |
| [run_pruning.py](02_论文复现与验证/01_SMoLK论文复现/run_pruning.py) | 删核与资源核算 |
| [run_smoke.py](02_论文复现与验证/01_SMoLK论文复现/run_smoke.py) | 12 核模型与最小计算验证 |

## 第二阶段：接入自己的心率流程

| 文件 | 职责 |
|---|---|
| [analyze_smolk_as_second_stage_v0.py](02_论文复现与验证/02_接入自己基线后的验证/01_代码/analyze_smolk_as_second_stage_v0.py) | 二阶段筛选探索 |
| [compare_smolk_hr_gating_v0.py](02_论文复现与验证/02_接入自己基线后的验证/01_代码/compare_smolk_hr_gating_v0.py) | 有效 v1 心率筛选主对照 |

## 第三阶段：波形重建

| 文件 | 职责 |
|---|---|
| [acc_preprocessing_v1.py](03_波形重建预研究/01_代码/acc_preprocessing_v1.py) | ACC 常量响应校正 |
| [evaluate_ppg_reconstruction_acc_consistency_v1.py](03_波形重建预研究/01_代码/evaluate_ppg_reconstruction_acc_consistency_v1.py) | 历史回放与固定验收 |
| [evaluate_ppg_reconstruction_v2.py](03_波形重建预研究/01_代码/evaluate_ppg_reconstruction_v2.py) | V2 波形/HR/逐搏评价 |
| [evaluate_ppg_reconstruction_v3.py](03_波形重建预研究/01_代码/evaluate_ppg_reconstruction_v3.py) | V3 固定检查与对照 |
| [prepare_ppg_reconstruction_acc_consistency_v1.py](03_波形重建预研究/01_代码/prepare_ppg_reconstruction_acc_consistency_v1.py) | 保持污染不变、替换 ACC 特征 |
| [prepare_ppg_reconstruction_pairs_v2.py](03_波形重建预研究/01_代码/prepare_ppg_reconstruction_pairs_v2.py) | BIDMC 人工污染配对数据 |
| [prepare_ppg_reconstruction_pairs_v3.py](03_波形重建预研究/01_代码/prepare_ppg_reconstruction_pairs_v3.py) | E4 域参考配对数据 |
| [run_acc_preprocessing_regression_v1.py](03_波形重建预研究/01_代码/run_acc_preprocessing_regression_v1.py) | 固定 V3 的 ACC 修正回归 |
| [run_ppg_acc_waveform_reconstruction_v0.py](03_波形重建预研究/01_代码/run_ppg_acc_waveform_reconstruction_v0.py) | ACC-NLMS 与 SMoLK 门控 |
| [run_ppg_acc_waveform_reconstruction_v1.py](03_波形重建预研究/01_代码/run_ppg_acc_waveform_reconstruction_v1.py) | 连续处理与双门控 |
| [test_acc_preprocessing_v1.py](03_波形重建预研究/01_代码/test_acc_preprocessing_v1.py) | 原有 ACC 数值测试 |
| [train_ppg_reconstruction_acc_consistency_v1.py](03_波形重建预研究/01_代码/train_ppg_reconstruction_acc_consistency_v1.py) | ACC 一致性重训 |
| [train_ppg_reconstruction_v2.py](03_波形重建预研究/01_代码/train_ppg_reconstruction_v2.py) | ResidualPPG 定义与 V2 训练 |
| [train_ppg_reconstruction_v3.py](03_波形重建预研究/01_代码/train_ppg_reconstruction_v3.py) | V3 单轮训练 |

这些是历史研究实现，含固定采样率、划分和校验。SMoLK 只输出伪影分数；V0/V1 与 ResidualPPG 才修改波形。新增代码的公开许可尚未确定；SMoLK 来源与保留许可见 [第三方说明](../THIRD_PARTY.md)。
