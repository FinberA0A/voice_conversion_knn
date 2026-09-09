"""F0 分析与操控：把第一课的源-滤波器模型真正写成可执行的代码。

第一课你在 02_source_filter.png 上看到，一帧浊音的频谱可以拆成两部分：
密集的谐波（声源，由 F0 决定）和平滑的包络（声道，决定音色）。WORLD 声码器
做的就是把整段音频做这个分解，拆成三样东西：

    f0  声源的基频      —— 音高，浊音帧为正值，清音帧为 0
    sp  频谱包络        —— 声道形状，也就是音色
    ap  非周期性        —— 气声/噪声成分的占比

关键性质：这三者可以独立修改再重新合成。只改 f0 而不动 sp，音高变了
但共振峰位置不变，听起来还是同一个人；这和直接重采样变调有本质区别，
后者会把整个频谱一起拉伸，共振峰跟着跑，于是产生"花栗鼠"或"巨人"效果。

explore/03_pitch.py 会让你亲耳听到这个差别。
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw

SR = 16000
FRAME_PERIOD = 5.0  # ms，WORLD 的默认帧移，与 WavLM 的 20ms 无关，各算各的


def analyze(wav: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """源-滤波器分解。harvest 求 F0 比 dio 慢但准得多，stonemask 再精修一遍。"""
    x = np.ascontiguousarray(wav, dtype=np.float64)
    f0, t = pw.harvest(x, sr, f0_floor=65.0, f0_ceil=500.0, frame_period=FRAME_PERIOD)
    f0 = pw.stonemask(x, f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    return f0, sp, ap


def synthesize(f0: np.ndarray, sp: np.ndarray, ap: np.ndarray,
               sr: int = SR) -> np.ndarray:
    y = pw.synthesize(np.ascontiguousarray(f0, dtype=np.float64),
                      sp, ap, sr, FRAME_PERIOD)
    return y.astype(np.float32)


def log_f0_stats(f0: np.ndarray) -> tuple[float, float]:
    """只统计浊音帧。清音帧的 0 混进来会把均值拉垮，且 log(0) 是负无穷。"""
    voiced = f0[f0 > 0]
    if voiced.size < 10:
        return float("nan"), float("nan")
    lf = np.log(voiced)
    return float(lf.mean()), float(lf.std())


def speaker_f0_stats(audio_dir: str | Path, max_files: int = 40,
                     cache_dir: str | Path = "outputs/.cache") -> tuple[float, float]:
    """统计一个说话人的音域。要多取几个文件，单句的音域不足以代表一个人。"""
    audio_dir = Path(audio_dir)
    cache = Path(cache_dir) / f"f0stats_{audio_dir.name}_{max_files}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        return d["mu"], d["sigma"]

    files = sorted(p for p in audio_dir.rglob("*")
                   if p.suffix.lower() in (".wav", ".flac", ".mp3"))[:max_files]
    if not files:
        raise FileNotFoundError(f"{audio_dir} 下没有音频")

    pool = []
    for p in files:
        wav, _ = librosa.load(str(p), sr=SR, mono=True)
        f0, _, _ = analyze(wav)
        pool.append(f0[f0 > 0])

    lf = np.log(np.concatenate(pool))
    mu, sigma = float(lf.mean()), float(lf.std())
    print(f"[f0 stats] {audio_dir.name}: {len(files)} 个文件, "
          f"中位音高 {np.exp(mu):.1f}Hz, log 标准差 {sigma:.3f}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"mu": mu, "sigma": sigma}))
    return mu, sigma


def shift_semitones(f0: np.ndarray, semitones: float) -> np.ndarray:
    """整体平移。清音帧的 0 必须原样保留 —— 给它赋一个正的 F0
    等于凭空把无声的辅音变成浊音，合成出来是持续的嗡嗡声。"""
    out = f0.copy()
    out[out > 0] *= 2.0 ** (semitones / 12.0)
    return out


def transfer_stats(f0: np.ndarray, src: tuple[float, float],
                   tgt: tuple[float, float]) -> np.ndarray:
    """log 域的均值方差迁移，第一课那个公式：

        log F0_out = (log F0_src - mu_src) / sigma_src * sigma_tgt + mu_tgt

    为什么在 log 域做？因为音高的感知是对数的 —— 100→200Hz 和 200→400Hz
    都是一个八度，听起来是"同样大"的变化，线性域的均值方差没有感知意义。
    sigma 的迁移会连语调起伏的幅度一起缩放到目标说话人的习惯范围。
    """
    mu_s, sig_s = src
    mu_t, sig_t = tgt
    out = f0.copy()
    v = out > 0
    if not v.any() or not np.isfinite([mu_s, sig_s, mu_t, sig_t]).all():
        return out
    out[v] = np.exp((np.log(out[v]) - mu_s) / sig_s * sig_t + mu_t)
    return out


def _fill_unvoiced(f0: np.ndarray) -> np.ndarray | None:
    """把清音帧的 0 用两侧浊音插值补上，得到一条连续曲线。

    只在做轮廓迁移时用：我们需要在"输出的浊音位置"读取"源的语调值"，
    而两者的清音分布并不一致，不补洞就会取到 0。
    """
    v = f0 > 0
    if v.sum() < 2:
        return None
    idx = np.arange(len(f0))
    return np.interp(idx, idx[v], f0[v])


def apply_source_contour(f0_out: np.ndarray, f0_src: np.ndarray,
                         src_stats: tuple[float, float],
                         tgt_stats: tuple[float, float],
                         source_voicing: bool = True) -> np.ndarray:
    """用源的语调轮廓重写输出的 F0，同时把音域迁移到目标。

    解决的问题：层号越深，检索出的音高越随机，语调飘忽。既然内容和音色
    kNN 都处理得不错，音高干脆不让它管，直接从源那边搬。

    source_voicing 决定哪些帧算浊音。默认跟随源，因为**浊清是内容属性而非
    音色属性**：/s/ 在谁口中都是清音，/a/ 在谁口中都是浊音。既然内容由源
    决定，浊清分布就该由源决定。用 kNN 输出的判定会两头漏——源浊输出清的
    帧丢掉语调，源清输出浊的帧写进插值出来的假值。
    """
    src_curve = _fill_unvoiced(f0_src)
    if src_curve is None:
        return f0_out

    # 两段音频长度不同（声码器会短 20ms，各自 F0 帧数也不同），按比例重采样
    old = np.linspace(0.0, 1.0, len(src_curve))
    new = np.linspace(0.0, 1.0, len(f0_out))
    aligned = np.interp(new, old, src_curve)

    shaped = transfer_stats(aligned, src_stats, tgt_stats)
    if source_voicing:
        v = np.interp(new, old, (f0_src > 0).astype(np.float64)) > 0.5
    else:
        v = f0_out > 0

    out = np.zeros_like(f0_out)
    out[v] = shaped[v]
    return out
