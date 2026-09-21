# 研究资料快照 · 2026-09-21

为现有 35 个研究脚本补充 198 份既有实验材料，共 6 个 ZIP、约 209.51 MiB。仓库保持 Private，本次没有新训练或改写历史结果。

- 01：自主基线资料，61 文件。
- 02：SMoLK 复现、映射、剪枝、固定回放与有效心率对照，35 文件。
- 03：波形重建配置、权重、结果、固定样例与配对元数据，93 文件。
- 04–06：V2、V3、ACC 一致性的 train/validation/test，9 个 NPZ。
- 每包另附 DATA_SOURCES.md，保留数据来源、署名与变换说明。

下载后先用 ARTIFACTS_SHA256SUMS 校验；ARTIFACT_CONTENTS.tsv 列出全部 ZIP 成员及其 SHA256。恢复路径、任务依赖和防覆盖说明见 [ARTIFACTS.md](https://github.com/lls-777/ppg-motion-artifact-lab/blob/main/ARTIFACTS.md)。

已核查源码副本一致、配对数组符合配置记录、固定 SMoLK 回放输出一致。尚未完成独立新环境和全部实验复跑。

必须同时保留的历史缺口：两份旧 V2 生成/评价源码快照未找到；第一阶段部分历史明细输出未找到。已有摘要不冒充缺失明细，新运行结果不补写成历史记录。详见 [REPRODUCTION_AUDIT.md](https://github.com/lls-777/ppg-motion-artifact-lab/blob/main/REPRODUCTION_AUDIT.md)。

完整官方原始数据与作者完整处理数组按原来源获取。材料补充不改变“真实运动可靠逐搏恢复尚未验证”的结论。
