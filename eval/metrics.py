"""客观评估：别只靠耳朵，但也别盲信数字。

VC 的质量是三个互相拉扯的维度，只看其中一个一定会调偏：
    1. 内容保持 —— 转换后还能听清说的是什么吗？用 ASR 转写比对
    2. 音色相似 —— 像目标角色吗？用说话人验证模型算嵌入余弦相似度
    3. 韵律保持 —— 语调还是源说话人的抑扬吗？用 F0 曲线的相关系数

两条从踩坑里换来的原则，写在这里免得再犯：

    一、指标必须有参照点。孤立的 speaker_cosine=0.61 什么也说明不了，
        必须同时算出"不转换时是多少"(下界) 和"目标说话人自己是多少"(上界)，
        才知道它落在这段区间的哪个位置。

    二、比对文本前必须正规化。ASR 把 "one two three" 转写成 "1 2 3" 时内容
        完全正确，但逐字符比会得到 0.8 的错误率，把好结果误判成灾难。

用法:
    python eval/metrics.py --source data/source/me.wav \
                           --converted outputs/converted.wav \
                           --target-dir data/target/LJspeech
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import librosa
import numpy as np

SR = 16000
HOP = 320

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def _num_to_en(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (" " + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        rest = _num_to_en(n % 100) if n % 100 else ""
        return f"{_ONES[n // 100]} hundred {rest}".strip()
    return str(n)


def normalize_text(text: str) -> tuple[str, bool]:
    """返回 (可比对的文本, 是否中文)。

    中文按字切分算 CER，英文按词切分算 WER —— 英文按字符算会把一个词的
    小拼写差异放大成多个错误，量纲和中文完全不同，两者不能共用阈值。
    """
    is_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))
    text = text.lower()
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
    # 数字按位读，这样源的 "9-1-1" 和转写的 "911" 会归一到同一串。
    # 代价是 "2008" 变成 "two zero zero eight" 而人念的是 "two thousand eight"。
    # 文本正规化没有完美解，所以下面必须配合差异明细人工审计。
    text = re.sub(r"\d+", lambda m: " " + " ".join(_ONES[int(d)] for d in m.group()) + " ",
                  text)
    text = " ".join(text.split())
    return (" ".join(text.replace(" ", "")) if is_chinese else text), is_chinese


def print_diff(ref: str, hyp: str, limit: int = 15) -> None:
    """逐词对齐后打印差异，让 WER 可审计。

    这一步不能省：正规化规则再怎么修补都会有边界情况，一个数字格式差异
    就能伪造出两个"错误"，把 layer6 的完美结果压成和 layer4 一样的分数。
    与其追求完美的正规化，不如让指标可以被人眼复核。
    """
    import jiwer

    out = jiwer.process_words(ref, hyp)
    r, h = ref.split(), hyp.split()
    chunks = [c for c in out.alignments[0] if c.type != "equal"]
    if not chunks:
        print("       差异明细 : 完全一致")
        return
    print(f"       差异明细 ({len(chunks)} 处):")
    for c in chunks[:limit]:
        rw = " ".join(r[c.ref_start_idx:c.ref_end_idx]) or "∅"
        hw = " ".join(h[c.hyp_start_idx:c.hyp_end_idx]) or "∅"
        print(f"         {c.type:<11} {rw}  ->  {hw}")
    if len(chunks) > limit:
        print(f"         ... 另有 {len(chunks) - limit} 处")


def content_preservation(src_path: str, conv_path: str) -> dict:
    """用同一个 ASR 转写源和转换结果，源的转写就是参考文本，不需要人工标注。"""
    try:
        from faster_whisper import WhisperModel
        import jiwer
    except ImportError:
        print("[skip] 未安装 faster-whisper / jiwer")
        return {}

    model = WhisperModel("small", device="cuda", compute_type="float16")

    def transcribe(path: str) -> str:
        segments, _ = model.transcribe(path, beam_size=5)
        return "".join(s.text for s in segments).strip()

    raw_ref, raw_hyp = transcribe(src_path), transcribe(conv_path)
    ref, is_cn = normalize_text(raw_ref)
    hyp, _ = normalize_text(raw_hyp)

    print(f"\n[内容] 源转写   : {raw_ref}")
    print(f"       转换后   : {raw_hyp}")
    if not ref:
        return {}
    print_diff(ref, hyp)
    return {("cer" if is_cn else "wer"): float(jiwer.wer(ref, hyp))}


def _embed_fn(device: str = "cuda:0"):
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="outputs/.speechbrain",
        run_opts={"device": device},  # 必须写 cuda:0，写 cuda 会触发解析警告
    )

    def embed(path: str):
        wav, _ = librosa.load(path, sr=SR, mono=True)
        e = enc.encode_batch(torch.from_numpy(wav).unsqueeze(0).to(device))
        return torch.nn.functional.normalize(e.squeeze(), dim=-1).cpu()

    return embed


def speaker_similarity(src_path: str, conv_path: str, target_dir: str) -> dict:
    """算三个数才有意义：转换结果、不转换的下界、目标说话人自身的上界。

    完成度 = (转换后 - 源) / (目标自身 - 源)
    接近 0 说明白转了，接近 1 说明音色已经到位。只有这个比值可以跨实验比较。
    """
    try:
        import torch
        embed = _embed_fn()
    except ImportError:
        print("[skip] 未安装 speechbrain")
        return {}

    targets = sorted(Path(target_dir).rglob("*.wav"))[:30]
    if len(targets) < 4:
        print("[skip] 目标目录音频太少")
        return {}

    embs = torch.stack([embed(str(p)) for p in targets])
    # 用前一半建参考、后一半算上界，避免同一批音频既当参考又当被测对象
    half = len(embs) // 2
    ref = torch.nn.functional.normalize(embs[:half].mean(0), dim=-1)
    upper = float(torch.stack([e @ ref for e in embs[half:]]).mean())
    lower = float(embed(src_path) @ ref)
    score = float(embed(conv_path) @ ref)

    denom = upper - lower
    return {
        "spk_converted": score,
        "spk_source_baseline": lower,
        "spk_target_ceiling": upper,
        "spk_completion": (score - lower) / denom if abs(denom) > 1e-6 else float("nan"),
    }


def f0_consistency(src_path: str, conv_path: str, plot: str | None = None) -> dict:
    """相关系数看语调"形状"，semitone shift 看整体音高被搬动了多少。

    两者必须分开看：knn-vc 不显式建模 F0，但特征替换会连带搬走音高，
    于是可能出现"形状跟得很好、整体高了大半个八度"这种情况。
    """
    def f0(path: str):
        wav, _ = librosa.load(path, sr=SR, mono=True)
        f, flag, _ = librosa.pyin(wav, fmin=65, fmax=400, sr=SR, hop_length=HOP)
        return f, flag

    f_src, v_src = f0(src_path)
    f_con, v_con = f0(conv_path)

    # 声码器输出比输入短约 20ms（无 padding 卷积的尾部截断），从头对齐后截断即可
    n = min(len(f_src), len(f_con))
    both = v_src[:n] & v_con[:n] & ~np.isnan(f_src[:n]) & ~np.isnan(f_con[:n])
    print(f"\n[F0] 可比对的浊音帧 {int(both.sum())} / {n}")
    if both.sum() < 20:
        print("     可比帧太少，下面的 F0 数字不可信，换一段更连贯的素材")
        return {}

    a, b = np.log2(f_src[:n][both]), np.log2(f_con[:n][both])
    shift = float(np.median(b - a) * 12)

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(n) * HOP / SR
        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        ax.plot(t, f_src[:n], label="source", lw=1.5)
        ax.plot(t, f_con[:n], label="converted", lw=1.5)
        ax.set(xlabel="time (s)", ylabel="Hz",
               title=f"F0 contours (median shift {shift:+.1f} semitones)")
        ax.legend()
        Path(plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot, dpi=120)
        plt.close(fig)
        print(f"     F0 对比图 -> {plot}")

    return {
        "f0_corr": float(np.corrcoef(a, b)[0, 1]),
        "f0_shift_semitone": shift,
        "f0_rmse_semitone": float(np.sqrt(np.mean((a - b) ** 2)) * 12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--converted", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--plot", default=None, help="F0 对比图输出路径")
    args = parser.parse_args()

    plot = args.plot or f"outputs/f0_{Path(args.converted).stem}.png"

    results = {}
    results.update(f0_consistency(args.source, args.converted, plot))
    results.update(speaker_similarity(args.source, args.converted, args.target_dir))
    results.update(content_preservation(args.source, args.converted))

    print("\n=== 评估结果 ===")
    for k, v in results.items():
        print(f"{k:>22}: {v:.4f}")
    print("\n读法:")
    print("  spk_completion   0=完全没转过去, 1=达到目标说话人自身水平")
    print("  f0_shift         整体音高搬动量, 12 = 一个八度")
    print("  f0_corr          >0.8 语调跟随良好; 输出崩坏时这个数不可信")
    print("  wer/cer          <0.15 内容清晰; 务必对照上面的原始转写一起看")


if __name__ == "__main__":
    main()
