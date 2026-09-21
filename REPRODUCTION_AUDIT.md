# 复现材料核查

[返回首页](README.md) · [资料下载与恢复](ARTIFACTS.md)

核查日期：2026-09-21。以下区分文件完整性、计算回放和历史版本追溯；不把文件齐备当作新环境完整复现。

## 已确认

- 三个研究阶段共 35 个 Python 文件，上传副本与本地源码逐字节一致，没有漏传这三个阶段的 Python 源文件。
- 补充包保留 198 份既有文件：基线 61、论文 35、重建说明/结果/元数据 93、配对 NPZ 9；每包另附同一份 DATA_SOURCES.md。
- 六个 ZIP 在内存生成并完成 CRC、成员列表与解压字节核查。资料清单逐文件记录 SHA256；原文件保持不变。
- 九个大型配对 NPZ 已实际重新计算 SHA256，全部符合其 config.json 的记录。
- 基线冻结 JSON 的 SHA256 为 ff3b3aba79e0de9c8931fdd476ac3b6628ebc3e144dc82f572277bd08ed9477e，与当前验证代码的期望一致。
- 重建来源链抽查的 56 项代码/权重/config/manifest/结果校验中，54 项匹配，2 项差异列于下节；这个计数不是模型性能。
- 现有环境完成 35 源码语法检查、8 项 ACC 测试、跨阶段导入、SMoLK/ResidualPPG 参数与前向接口检查。
- 用本地 NumPy 1.22.4、SciPy 1.9.3 执行固定 SMoLK 回放：原始分数最大差 0、SG 分数最大差 0、二值掩码差异 0。
- 未重新训练；未完成独立新环境安装或全部端到端实验。历史验收未通过的结论保持原样。

## 两份旧 V2 源码快照缺失

| 历史记录 | 记录中的源码 SHA256 | 当前源码 SHA256 |
|---|---|---|
| pairs_v2/config.json 的 prepare_ppg_reconstruction_pairs_v2.py | ce9487fcca91f04a5aa26d305d0b09765c0540dd364970efbf7f5ba599f95642 | c01d679fbe0a740b38649670d03423d9459a1abfdea92f484a8dc9cefe3766ac |
| V2/evaluation/summary.json 的 evaluate_ppg_reconstruction_v2.py | f7b27424ae175444a4c42c9407306c4e94fae5681005cf1019d38e047196499a | 8c19e94c3652b40b43f9c8b04b3aa47a47b8fbce3a44918e0deed1e73961c350 |

在研究工程内未找到对应旧快照。不能断言差异只来自目录调整，也不能重写记录中的哈希让它“匹配”。

V3 与 ACC 后续链引用的是当前较新 helper 的哈希，相应校验一致；这两处 V2 历史差异不等于整个后续链必然无法运行。

## 第一阶段未找到的历史输出

在 acc_aware_sqi_v1_validation 中，未找到现有脚本所写的：

- S11_S15_acc_aware_sqi_v1_validation_windows.csv
- subject_metrics.csv
- activity_metrics.csv

现有 subject_summary.csv、activity_summary.csv 与验证 JSON 是历史摘要，原样保留，不改名冒充上述文件。

在 candidate_peak_recoverability_v0 中，仅找到报告，未找到：

- failure_window_results.csv、candidate_peak_results.csv、scope_summary.csv
- subject_summary.csv、activity_summary.csv、candidate_peak_recoverability_v0.json

这些是现有脚本的输出，不是冻结验证的前置输入。按当前源码可另行重算；新的结果必须注明新的运行时间与源码版本，不能补写成原历史记录。

## 其他边界

- 已作废的 smolk_hr_comparison_v0 只保留 INVALID.md；不打包其五份无效数值/图片。当前有效主对照是 v1。
- V0/V1 为传统处理，没有神经网络训练权重；并非权重漏传。
- V3 验收为 criteria_not_met（177 项中 41 项未通过）；ACC 一致性为 development_checks_not_met（170 项中 21 项未通过）。state=complete 只表示计算完成。
- UniCardio/SSA 未找到可独立追溯的完整工件，未据此增加复现成果声明。
- 历史 Markdown/JSON 中的本机路径和旧环境命令保留为原始记录；跨电脑操作使用 RUNNING.md 与 ARTIFACTS.md。
- 全部资料仍不能提供真实运动的同步干净 PPG 真值，也不构成 HRV 或自有硬件验证。
