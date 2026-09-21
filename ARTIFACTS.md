# 实验资料下载与恢复

[返回首页](README.md) · [运行说明](RUNNING.md) · [历史缺口](REPRODUCTION_AUDIT.md)

资料入口：[私有 Release：研究资料快照 2026-09-21](https://github.com/lls-777/ppg-motion-artifact-lab/releases/tag/research-artifacts-2026-09-21)。需要有仓库访问权限的 GitHub 账号。源码继续保存在 research；大文件作为同一私有仓库的 Release 附件保存。GitHub 官方建议用 Release 分发这类大文件，见 [文件大小与发布附件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

## 下载内容

| 附件 | 既有文件数 | 压缩大小 | 内容 |
|---|---:|---:|---|
| 01-baseline-evidence.zip | 61 | 23.73 MiB | 基线冻结规则、特征/逐窗表、统计、图和历史说明 |
| 02-smolk-evidence.zip | 35 | 1.70 MiB | SMoLK 权重/checkpoint、映射、剪枝、固定回放、有效 HR 对照 |
| 03-reconstruction-evidence.zip | 93 | 14.63 MiB | V0–V3、ACC 修正/一致性结果、权重、配对元数据和小型样例 |
| 04-pairs-v2.zip | 3 | 58.82 MiB | V2 的 train/validation/test 三个 NPZ |
| 05-pairs-v3.zip | 3 | 55.31 MiB | V3 的 train/validation/test 三个 NPZ |
| 06-pairs-acc-consistency-v1.zip | 3 | 55.31 MiB | ACC 一致性的 train/validation/test 三个 NPZ |

六包共 198 份既有文件；每包另附 DATA_SOURCES.md。01–03 号含 189 份配置、权重、表格、图片及历史说明；04–06 号补齐九个配对数组。可按任务选择下载；只下载元数据而缺 NPZ 不能训练。

附件校验见 [ARTIFACTS_SHA256SUMS](ARTIFACTS_SHA256SUMS)，全部成员及其校验值见 [ARTIFACT_CONTENTS.tsv](ARTIFACT_CONTENTS.tsv)。每个历史成员逐字节保留；没有修改模型、历史成绩、冻结 JSON 或源码哈希。

官方原始 DaLiA/BIDMC 数据不在包中。SMoLK 的完整作者数组由现有脚本按固定提交获取；保存的固定回放只含一段 DaLiA 派生输入及模型输出，来源见 [DATA_SOURCES.md](DATA_SOURCES.md)。

## 先校验，再查看历史

将六个 ZIP 和校验文件下载到本地 release-assets 目录；这份目录被 Git 忽略。以下在仓库根目录执行：

~~~bash
cd release-assets
shasum -a 256 -c ../ARTIFACTS_SHA256SUMS
cd ..
mkdir -p evidence-view
unzip -n release-assets/01-baseline-evidence.zip -d evidence-view
unzip -n release-assets/02-smolk-evidence.zip -d evidence-view
unzip -n release-assets/03-reconstruction-evidence.zip -d evidence-view
unzip -n release-assets/04-pairs-v2.zip -d evidence-view
unzip -n release-assets/05-pairs-v3.zip -d evidence-view
unzip -n release-assets/06-pairs-acc-consistency-v1.zip -d evidence-view
~~~

上面使用 macOS 的 shasum；Linux 可使用 sha256sum -c。下载部分包时只核验对应行，不把未下载附件的报错解释为已下载文件损坏。unzip -n 保留同名现有文件；它不是完整性验证，目录已存在时仍需对照成员清单。

历史文件位于 evidence-view/research/三个阶段目录。下表路径相对于该 research：

| 要查看的证据 | 入口 |
|---|---|
| 基线冻结规则 | 01_自主基线研究/02_结果/acc_aware_sqi_v1_design/acc_aware_sqi_v1_frozen_design.json |
| 基线验证摘要 | 01_自主基线研究/02_结果/acc_aware_sqi_v1_validation/acc_aware_sqi_v1_validation_result.json |
| SMoLK 完整复现 | 02_论文复现与验证/01_SMoLK论文复现/outputs/full_seed0/result.json |
| 有效 HR 主对照 | 02_论文复现与验证/02_接入自己基线后的验证/02_结果/smolk_hr_comparison_v1/summary.json |
| V2/V3 评价 | 03_波形重建预研究/02_结果/V2/evaluation/summary.json；同级 V3/evaluation/summary.json |
| ACC 修正诊断 | 03_波形重建预研究/02_结果/ACC预处理修正V1/summary.json |
| 最新固定验收 | 03_波形重建预研究/02_结果/ACC训练推理一致性V1/evaluation/acceptance.json |

旧说明中的 file:///Users/...、Obsidian 链接及旧环境命令是历史记录，可能在其他电脑不可用；不按旧绝对路径创建目录。当前源码使用 research 下的相对阶段结构。

## 无需原始数据的固定回放

安装 RUNNING.md 所述依赖后，02 号包即可支持已保存 SMoLK 核心的只读回放：

~~~bash
python -B research/02_论文复现与验证/01_SMoLK论文复现/deployment_numpy_runtime.py evidence-view/research/02_论文复现与验证/01_SMoLK论文复现/outputs/deployment_replay_v0
~~~

程序重新计算一段输入的概率、151 点 SG 与掩码，和保存数组比较，只打印结果。当前原环境三项差异均为 0；跨库版本/硬件的逐位一致性需实际检查。本回放不包含原始带通、训练或完整测试集性能复算。

## 重新计算使用独立副本

查看历史与重新训练/评价分开。准备一个独立仓库副本，只把该任务需要的输入依赖从 evidence-view/research 复制到新副本的 research，保持本次目标输出目录不存在。

不要先把六包全量恢复到运行副本再直接运行全部入口：已有 result、checkpoint、state 或输出目录会触发拒绝覆盖/续训；删除一个 result.json 也不等于重新训练。

| 任务 | 需要恢复或准备 | 必须不存在的目标目录 |
|---|---|---|
| S1 心率基线 | 官方 S1.pkl | 第一阶段 02_结果/hr_baseline |
| 冻结 SQI＋ACC 验证 | 01 号的冻结设计、设计/PPG 窗口表；官方 S11–S15.pkl | acc_aware_sqi_v1_validation |
| SMoLK 从头训练 | 固定作者数组（原脚本下载）；不恢复旧 checkpoint | outputs/full_seed0 整个目录 |
| 心率主对照 | 01 号依赖、02 号权重/映射；官方 S11–S15.pkl | smolk_hr_comparison_v1 |
| V2 重新训练 | 03 号中 V2 配对 config/state/manifest，加 04 号三个 NPZ | V2/training |
| V3 重新评价 | 02 号映射，03 号依赖权重和历史 V2 结果，05 号数组，官方 S14/S15.pkl | V3/evaluation |
| ACC 固定权重修正 | 03 号原 V3 配置/权重/逐窗表及固定样例；官方 S14/S15.pkl | ACC预处理修正V1 |
| ACC 一致性评价 | 03 号历史依赖和两套权重，05/06 号数组；官方 S14/S15.pkl | ACC训练推理一致性V1/evaluation |

表格是依赖与输出边界，命令参数仍以 RUNNING.md 和各脚本定义为准。V0/V1 不带 --write-outputs 仍会执行读数与计算，不是轻量 dry-run。历史数据文件要求完整配套，不能只复制 config/state 却漏掉 NPZ。

如果需要从头生成配对数组，用新的空配对目录和官方数据，保留生成记录；它是新的运行，不能拿新字节覆盖原历史 config/hash。V2 两份旧源码快照缺失，具体限制见 REPRODUCTION_AUDIT.md。

## 数据与许可

每包保留 [数据署名和变换说明](DATA_SOURCES.md)。官方原始全集、作者完整处理数组、虚拟环境、字体缓存、失效旧对照的数值文件不打包；旧对照 INVALID.md 保留。主工程和已发布源码保持不变。

资料包扩充的是证据与输入，不改变研究结论：当前仍未验证真实运动中的可靠逐搏恢复。仓库保持私有，公开范围另行确认。
