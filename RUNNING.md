# 运行说明

[返回首页](README.md) · [全部源码](research/README.md)

## 本版提供什么

35 个 Python 脚本按原研究目录逐字节复制；保留模块边界、超参数、失败保护和历史校验。三阶段中的不同方法没有合并成一个新模型。原始数据、处理后数组、权重、生成的 CSV/JSON 和虚拟环境不随仓库分发。

当前完成源码静态编译和依赖/路径审计，尚未验证干净新环境安装，也未在本版目录重新训练。以下是已有代码支持的命令与前置条件，不是“所有实验已在新机器验证通过”的声明。

## 环境

原实验记录的环境为 Python 3.9.6，NumPy 1.22.4、SciPy 1.9.3、PyTorch 2.1.1、scikit-learn 1.3.2、Matplotlib 3.7.5。requirements.txt 固定这五项直接依赖；它不是跨平台的完整依赖锁。使用同代 Python 环境，其他 Python/平台组合需单独验证。

以下示例适用于 macOS/Linux，从仓库根目录执行；python3 应指向选定的 Python 3.9 环境：

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

代码主要运行在 CPU。不要只在某一子目录复制单个脚本；跨阶段导入依赖下列相对布局：

~~~text
research/
  01_自主基线研究/01_代码/
  02_论文复现与验证/
    01_SMoLK论文复现/
    02_接入自己基线后的验证/01_代码/
  03_波形重建预研究/01_代码/
  90_共享原始数据/                 # 数据由使用者准备，不在仓库
~~~

输出仍写在各阶段的 02_结果、SMoLK 的 outputs，以及重建阶段的 03_重建训练数据。忽略规则覆盖这些运行目录，4 张展示图单独保存在 assets。

## 先运行无需原始数据的检查

ACC 数值测试直接复用原测试文件，不下载数据，也不训练模型：

~~~bash
python -B -m unittest discover -s research/03_波形重建预研究/01_代码 -p test_acc_preprocessing_v1.py -v
~~~

该测试检查常量响应、历史模式及特征等数值性质；测试通过不代表真实逐搏重建成功。

## 第一阶段：心率基线及 SQI＋ACC

从 [UCI PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg%2Bdalia) 获取数据，保留官方目录。S1 的预期位置是：

~~~text
research/90_共享原始数据/ppg_dalia/PPG_FieldStudy/S1/S1.pkl
~~~

pickle 只应读取从可信官方来源取得的文件。主要字段如下：

| 字段 | 形状/采样率 | 用途 |
|---|---|---|
| signal/wrist/BVP | 单路，可展平为 N 点；64 Hz | 心率与质量特征输入 |
| label | 窗口级 ECG 导出 HR | 8 秒窗、2 秒步长的评价参考 |
| signal/wrist/ACC | 三轴；32 Hz；原实现按 g 处理 | 运动特征 |
| activity | 4 Hz | 活动分组 |
| rpeaks | ECG 700 Hz 的 R 峰样点位置 | 后续重建的逐搏评价尝试 |

HR 基线只读取 BVP 和 label。标签数需等于 1 + floor((N-512)/128)；至少需要 512 个 PPG 点。

~~~bash
python -B research/01_自主基线研究/01_代码/inspect_s1_ppg_dalia.py
python -B research/01_自主基线研究/01_代码/ppg_hr_baseline.py
~~~

基线是去均值/线性趋势、Hann 窗、4096 点 FFT、30–220 bpm 主峰搜索及插值；不是带通去噪或波形重建。它固定 64 Hz、8 秒窗、2 秒步长，输出窗口表、指标 JSON 和两张 S1 场景图。场景为 120–180 秒、850–910 秒，因此 main 入口使用完整 S1，不能拿短合成信号代替。

基线等早期脚本没有 argparse，不能假定 --help 是只读入口；直接运行会写结果，部分早期脚本可能覆盖同名输出。请在新的仓库副本中运行，保留需追溯的旧结果。

完整 SQI＋ACC 的研究顺序是：

~~~text
S1–S15 原始数据
→ analyze_all_subjects_ppg_sqi.py：逐窗 PPG 特征
→ evaluate_ppg_only_sqi_v0.py：设计 PPG SQI 与状态
→ build_acc_motion_features_v0.py：ACC 运动特征
→ evaluate_acc_aware_sqi_v1.py：S1–S10 规则设计
→ validate_acc_aware_sqi_v1.py：S11–S15 冻结验证
~~~

这些脚本依赖前一步生成的表格。大部分显式使用 --write-outputs；最终冻结验证还需要 --run-validation。先阅读各脚本 main/参数定义再执行，不把上述顺序当作可跳过中间结果的单命令。

冻结验证包含固定受试者/窗口数和规则 JSON 哈希。若缺少原冻结规则或生成的字节不同，保留失败状态，先检查输入、版本和文件来源，不修改哈希常量绕过。当前仓库提供该流程源码，没有附带原冻结设计文件。

## 第二阶段：SMoLK 单种子复现

两条命令互相独立：

~~~bash
python -B research/02_论文复现与验证/01_SMoLK论文复现/run_smoke.py
python -B research/02_论文复现与验证/01_SMoLK论文复现/run_full.py
~~~

- smoke：训练前 64 段、测试前 8 段，32 次参数更新；下载固定作者提交的 NPY 前缀，输出 smoke_run.json 和 segment_learning.png，用于理解计算链。
- full：下载固定作者提交的 8 个完整 NPY，合计约 214 MiB；训练 DaLiA 的 3436 段，512 次更新，评价 DaLiA/WESAD/TROIKA。下载时校验大小、形状、dtype 和固定 Git blob SHA-1。
- 每段单路 64 Hz、1920 点，逐段标准化；0 是干净标签，1 是伪影标签。12 个核、1432 个参数。
- 完整结果保存到该模块 outputs/full_seed0，包含 model_state.pt、result.json、summary.png 和 checkpoint.pt。
- 已有正式结果会被拒绝覆盖；full 仅从符合校验的自身 checkpoint 续训。不要加载不可信模型文件。
- 这是作者处理后数组的单种子复现；未包含从全部原始数据重建预处理或多种子统计的自动流程。

作者数据来源写在脚本中，固定提交为 4aa63d72d5fedb80bd5ba6a905dcc75f1cafd12a。本仓库不再分发这些数组；数据使用遵循各自来源要求。

后续 recover_dalia_mapping.py 需要原始 DaLiA 与发布数组，生成 outputs/dalia_mapping 的时间映射。下游 compare_smolk_hr_gating_v0.py 使用当前有效的 v1 输出目录，还需要第一阶段冻结验证结果及 SMoLK 权重。run_pruning.py 和 run_deployment_replay.py 依赖完整训练/剪枝输出；电脑端回放不代表已完成 MCU 部署。

## 第三阶段：波形重建

模型定义在 train_ppg_reconstruction_v2.py 的 ResidualPPG。输入形状为 (batch,5,512)：PPG＋ACC xyz＋模长；输出为输入 PPG 减去网络估计残差，共 5393 个参数。后续 V3 与 ACC 一致性训练复用该类。

V2 准备需要 DaLiA 的 S2–S13 原始 ACC；BIDMC 缺失文件会由脚本从固定 PhysioNet 地址下载并核验官方 SHA256。BIDMC 参考经筛选和带限，不能称为无噪声生理真值。

~~~bash
python -B research/03_波形重建预研究/01_代码/prepare_ppg_reconstruction_pairs_v2.py --write-outputs
python -B research/03_波形重建预研究/01_代码/train_ppg_reconstruction_v2.py --write-outputs
~~~

数据流：

~~~text
BIDMC 125 Hz PLETH → 64 Hz 带限参考 → 按患者划分
DaLiA 32 Hz ACC → 64 Hz → xyz＋模长、去趋势/标准化
→ identity、10/0/−5 dB 人工污染配对
→ train/validation/test NPZ＋manifest＋config＋state
→ 20 epoch、L1、验证集选权重
~~~

NPZ 的 x 为 float32 (n,5,512)，y 为 float32 (n,1,512)。患者和 ACC 供体按集合隔离。模型输出和目标共用污染输入的尺度，不在评价时重新缩放贴合参考。

V2 完整评价还需要 S14/S15 原始数据、上述第二阶段生成的 dalia_hr_window_mapping.csv，以及训练结果：

~~~bash
python -B research/03_波形重建预研究/01_代码/evaluate_ppg_reconstruction_v2.py --write-outputs
~~~

此入口同时做合成与真实评价，没有单独的“只评价配对数据”开关。缺少真实数据或映射时，不能把部分输出当作一次完整成功运行。它比较原输入、带通、ACC-NLMS、残差模型，并记录波形、HR 和有效逐搏评价条件。

V0/V1 是传统 NLMS 与门控；V2/V3 是残差网络。acc_preprocessing_v1.py 提供独立的常量响应校正；V2 历史处理保留原样，没有把修正悄悄混入原结果。

最新 ACC 一致性回放依赖原 V3 配对数据/config/manifest、旧训练配置和权重、ACC 修正回归 CSV/JSON、旧真实评价表，以及对应代码与库版本。源码全部收录，但这些历史工件未随包提供。不要改写历史 SHA256 或放宽验收阈值来强行运行；也不要把新训练输出当作原历史实验的精确复现。

## 结果如何解读

重新运行会产生新的运行记录。是否复现案例中的数字，应先核对数据/源码哈希、划分、版本、预处理和评价集合；源码存在或数值测试通过都不能代替这一步。

现有结论仍是：合成去噪有改善，真实运动中稳定可靠的逐搏恢复尚未验证，不能据此声明 HRV 或自有硬件效果。原始数据与历史工件的后续纳入，以及公开发布范围，另行确定。
