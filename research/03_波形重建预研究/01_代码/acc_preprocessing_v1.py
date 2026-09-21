"""ACC 重采样 V1：显式选择历史处理或单位常量响应校正。"""
from __future__ import annotations

import numpy as np
from scipy import signal

CORRECTED_MODE = "polyphase_dc_normalized_v1"


def resample_acc(acc, *, dc_normalize: bool):
    """完整连续三轴 ACC 从 32 Hz 到 64 Hz；调用后再切窗和提取特征。

    L 为同一 resample_poly 算子。L(acc)/L(ones) 保持常量输入，
    不保证任意动态信号无失真。历史回放必须显式传 False。
    """
    values = np.asarray(acc, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("ACC must have shape (n>=2, 3)")
    if not np.isfinite(values).all():
        raise ValueError("ACC contains nonfinite values")
    if not isinstance(dc_normalize, bool):
        raise TypeError("dc_normalize must be an explicit bool")
    output = signal.resample_poly(values, 2, 1, axis=0)
    if dc_normalize:
        response = signal.resample_poly(np.ones(len(values)), 2, 1)
        # 默认补零的记录边缘约为 0.5；不可误用 >0.5 排除合法边缘。
        if not np.isfinite(response).all() or np.any(response <= 0.1):
            raise ValueError("Unsafe constant-response denominator")
        output = output / response[:, None]
    if not np.isfinite(output).all():
        raise ValueError("Resampling produced nonfinite values")
    return output
