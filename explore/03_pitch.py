"""变调的两种做法，以及为什么源-滤波器模型不是纸上谈兵。

用法:
    python explore/03_pitch.py data/source/complicateSentenceVocals.wav --semitones 7

生成的三个音频请按顺序听:
    original.wav          原声
    shift_naive.wav       朴素变调 —— 整个频谱一起拉伸，共振峰跟着跑
    shift_world.wav       源-滤波器变调 —— 只动 F0，共振峰纹丝不动

朴素变调听起来像"花栗鼠"或"巨人"，因为共振峰位置编码的是声道的物理尺寸，
把它整体上移等于宣称这个人的口腔突然变小了。WORLD 的做法是先把声源和声道
分离，只把声源的基频改掉，声道原样保留，于是音高变了、人还是那个人。

图 02_envelope_compare.png 会把这件事画出来给你看。
"""

import argparse
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from knnvc.pitch import FRAME_PERIOD, SR, analyze, log_f0_stats, shift_semitones, synthesize


def plot_world_decomposition(wav: np.ndarray, out_dir: Path) -> None:
    f0, sp, ap = analyze(wav)
    t = np.arange(len(f0)) * FRAME_PERIOD / 1000

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True, sharex=True)

    axes[0].plot(t, np.where(f0 > 0, f0, np.nan), lw=1.5)
    axes[0].set(ylabel="Hz", title="f0 —— 声源基频（清音帧为 0，图上断开）")

    axes[1].imshow(np.log(sp + 1e-12).T, aspect="auto", origin="lower",
                   extent=[0, t[-1], 0, SR // 2], cmap="magma")
    axes[1].set(ylabel="Hz", title="sp —— 频谱包络，声道形状，音色就藏在这里")

    axes[2].imshow(ap.T, aspect="auto", origin="lower",
                   extent=[0, t[-1], 0, SR // 2], cmap="viridis")
    axes[2].set(xlabel="time (s)", ylabel="Hz",
                title="ap —— 非周期性，亮的地方是气声和摩擦音")

    fig.savefig(out_dir / "01_world_decomposition.png", dpi=120)
    plt.close(fig)


def compare_shift(wav: np.ndarray, semitones: float, out_dir: Path) -> None:
    f0, sp, ap = analyze(wav)
    world = synthesize(shift_semitones(f0, semitones), sp, ap)

    # librosa 的做法是时间拉伸再重采样，等价于把整个频谱轴按比例缩放
    naive = librosa.effects.pitch_shift(wav, sr=SR, n_steps=semitones)

    sf.write(out_dir / "original.wav", wav, SR)
    sf.write(out_dir / "shift_world.wav", world, SR)
    sf.write(out_dir / "shift_naive.wav", naive, SR)

    # 在同一个时间位置上比较三者的频谱包络
    rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=256)[0]
    ratio = float(np.argmax(rms)) / max(len(rms) - 1, 1)

    freqs = np.linspace(0, SR // 2, sp.shape[1])
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)

    for name, sig, style in [("original", wav, "-"),
                             ("world (f0 only)", world, "-"),
                             ("naive (whole spectrum)", naive, "--")]:
        _, s, _ = analyze(sig)
        idx = min(int(ratio * (s.shape[0] - 1)), s.shape[0] - 1)
        ax.plot(freqs, np.log(s[idx] + 1e-12), style, lw=2, label=name, alpha=0.85)

    ax.set(xlabel="frequency (Hz)", ylabel="log spectral envelope", xlim=(0, 5000),
           title=f"变调 {semitones:+g} 个半音后，共振峰去哪了")
    ax.legend()
    fig.savefig(out_dir / "02_envelope_compare.png", dpi=120)
    plt.close(fig)

    print("\n看 02_envelope_compare.png：")
    print("  实线 original 和 world 的峰值位置应当几乎重合 —— 音色没被动过")
    print("  虚线 naive 的峰整体平移了 —— 声道被'物理缩放'了，所以变声")

    for name, sig in [("原声", wav), ("world 变调", world), ("naive 变调", naive)]:
        f, _, _ = analyze(sig)
        mu, _ = log_f0_stats(f)
        print(f"  {name:<12} 中位音高 {np.exp(mu):6.1f} Hz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--semitones", type=float, default=7.0,
                        help="正数升调。7 个半音正好是实验里测到的男声->LJSpeech 的差距")
    args = parser.parse_args()

    out_dir = Path("outputs/pitch")
    out_dir.mkdir(parents=True, exist_ok=True)

    wav, _ = librosa.load(args.audio, sr=SR, mono=True)
    wav = librosa.util.normalize(wav) * 0.95

    plot_world_decomposition(wav, out_dir)
    compare_shift(wav, args.semitones, out_dir)
    print(f"\n完成，图和音频都在 {out_dir.resolve()}")


if __name__ == "__main__":
    main()
