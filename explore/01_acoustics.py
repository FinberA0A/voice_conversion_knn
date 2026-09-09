"""声学基础探索：把"源-滤波器模型"和时频分析亲手跑一遍。

用法:
    python explore/01_acoustics.py data/target/sample.wav

会在 outputs/acoustics/ 下生成几张图和两个音频文件，逐个看完你就理解了
后面 VC 模型里每一步在操作什么。
"""

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import numpy as np
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 16000
N_FFT = 1024
HOP = 320       # 20ms @16k，与 WavLM 的帧率对齐，方便后面对照
WIN = 800       # 50ms
N_MELS = 80


def load(path: str) -> np.ndarray:
    wav, _ = librosa.load(path, sr=SR, mono=True) #sr: 表示将一秒分成多少份进行取样, mono: 表示是否是单声道波形
    return librosa.util.normalize(wav) * 0.95 #normalize到-1和1的时候，相当于将振幅调整了

#单纯绘制了波形的样子和梅尔谱的情况，用于对声音有更深刻的理解
def plot_waveform_and_spectrograms(wav: np.ndarray, out_dir: Path) -> np.ndarray:
    linear = np.abs(librosa.stft(wav, n_fft=N_FFT, hop_length=HOP, win_length=WIN))
    mel = librosa.feature.melspectrogram(
        S=linear**2, sr=SR, n_mels=N_MELS, fmin=0, fmax=SR // 2
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)

    axes[0].plot(np.arange(len(wav)) / SR, wav, lw=0.5)
    axes[0].set(title="1. Raw waveform", xlabel="time (s)", ylabel="amplitude")

    img = librosa.display.specshow(
        librosa.amplitude_to_db(linear, ref=np.max),
        sr=SR, hop_length=HOP, x_axis="time", y_axis="linear", ax=axes[1],
    )
    axes[1].set(title=f"2. Linear STFT magnitude ({linear.shape[0]} freq bins)")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

    img = librosa.display.specshow(
        mel_db, sr=SR, hop_length=HOP, x_axis="time", y_axis="mel", ax=axes[2],
    )
    axes[2].set(title=f"3. Mel spectrogram ({N_MELS} bins) —— 压缩了 6 倍，感知上几乎无损")
    fig.colorbar(img, ax=axes[2], format="%+2.0f dB")

    fig.savefig(out_dir / "01_time_frequency.png", dpi=120)
    plt.close(fig)
    return linear


def plot_source_filter(wav: np.ndarray, out_dir: Path) -> None:
    """在单帧上把频谱拆成"包络(声道)"和"精细结构(声源)"。

    倒谱法：对 log 幅度谱做 IFFT 得到倒谱，低"quefrency"部分对应缓慢变化的
    频谱包络（声道共振），高部分对应密集的谐波峰（声带基频）。
    """
    rms = librosa.feature.rms(y=wav, frame_length=WIN, hop_length=HOP)[0]
    center = int(np.argmax(rms)) * HOP  # 挑能量最大的一帧，大概率是个元音 能量大表示这个wav的振幅大
    frame = wav[center: center + WIN] * np.hanning(WIN)
    #因为默认一块窗内的波形是循环的，所以需要乘以一个汉宁窗，使得窗的两端为0，中间为1，防止首尾相接的时候出问题

    #开始对取出来的帧进行处理，取出其中的包络（音色）和精细结构（声带谐波）
    spec = np.fft.rfft(frame, n=N_FFT)
    log_mag = np.log(np.abs(spec) + 1e-9)

    cepstrum = np.fft.irfft(log_mag)
    lifter = np.zeros_like(cepstrum)
    lifter[:30] = 1.0                  # 只保留低倒频，即平滑包络
    lifter[-29:] = 1.0
    envelope = np.fft.rfft(cepstrum * lifter).real

    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    #以上设计声学的倒谱等知识，可以参考https://blog.csdn.net/weixin_43914614/article/details/123901423
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.plot(freqs, log_mag, lw=0.8, alpha=0.6, label="full spectrum (harmonics = source/F0)")
    ax.plot(freqs, envelope, lw=2.2, label="cepstral envelope (vocal tract = timbre)")

    peaks = [i for i in range(2, len(envelope) - 2)
             if envelope[i] > envelope[i - 1] and envelope[i] > envelope[i + 1]]
    for k, p in enumerate(peaks[:4]):
        ax.axvline(freqs[p], color="r", ls="--", alpha=0.5)
        ax.text(freqs[p], envelope.max(), f"F{k+1}", color="r")

    ax.set(title="Source-Filter decomposition of one voiced frame",
           xlabel="frequency (Hz)", ylabel="log magnitude", xlim=(0, 5000))
    ax.legend()
    fig.savefig(out_dir / "02_source_filter.png", dpi=120)
    plt.close(fig)

    print(f"[source-filter] 前 4 个共振峰约在: "
          f"{[int(freqs[p]) for p in peaks[:4]]} Hz")

#f0是基频(foundamental frequency)，用于表示声音的音高，可以用于说话人音高迁移等操作
def analyse_f0(wav: np.ndarray, out_dir: Path) -> None:
    f0, voiced_flag, _ = librosa.pyin(
        wav, fmin=65, fmax=400, sr=SR, frame_length=WIN * 2, hop_length=HOP
    )
    voiced = f0[voiced_flag & ~np.isnan(f0)]
    log_f0 = np.log(voiced)

    print(f"[F0] 浊音帧占比 {voiced.size / f0.size:.1%}  "
          f"中位数 {np.median(voiced):.1f} Hz  "
          f"log-F0 均值 {log_f0.mean():.3f} 标准差 {log_f0.std():.3f}")
    print("     ^ 这两个统计量就是做说话人音高迁移时要用的 mu 和 sigma")

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    times = librosa.times_like(f0, sr=SR, hop_length=HOP)
    ax.plot(times, f0, lw=1.5)
    ax.set(title="F0 contour (gaps = unvoiced frames)", xlabel="time (s)", ylabel="Hz")
    fig.savefig(out_dir / "03_f0_contour.png", dpi=120)
    plt.close(fig)


def demo_phase_loss(wav: np.ndarray, linear: np.ndarray, out_dir: Path) -> None:
    """听一下"丢掉相位"的代价，这就是为什么我们需要神经声码器。"""
    #丢掉相位，那么各个谐波的相位就无法保持一致，导致听起来像是金属声
    gl = librosa.griffinlim(linear, hop_length=HOP, win_length=WIN, n_iter=32)
    sf.write(out_dir / "original.wav", wav, SR)
    sf.write(out_dir / "griffinlim_magnitude_only.wav", gl, SR)
    print("[phase] 对比听 outputs/acoustics/ 下的两个 wav："
          "后者只用幅度谱重建，那股机械的金属感就是相位丢失造成的")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="任意一段 wav/mp3，建议 3~10 秒的清晰人声")
    args = parser.parse_args()

    out_dir = Path("outputs/acoustics")
    out_dir.mkdir(parents=True, exist_ok=True)

    wav = load(args.audio)
    print(f"[load] {len(wav) / SR:.2f}s @ {SR}Hz -> "
          f"{1 + len(wav) // HOP} 帧 (每帧 {HOP / SR * 1000:.0f}ms)")

    linear = plot_waveform_and_spectrograms(wav, out_dir)
    plot_source_filter(wav, out_dir)
    analyse_f0(wav, out_dir)
    demo_phase_loss(wav, linear, out_dir)
    print(f"\n完成，图和音频都在 {out_dir.resolve()}")


if __name__ == "__main__":
    main()
