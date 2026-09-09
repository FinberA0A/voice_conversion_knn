"""端到端转换：源音频 -> WavLM 特征 -> kNN 替换 -> HiFi-GAN 合成波形。

用法:
    python -m knnvc.convert --source data/source/me.wav \
                            --target-dir data/target/character \
                            --topk 4 --out outputs/converted.wav

声码器为什么用别人的预训练权重？
    从零训一个 HiFi-GAN 需要几百小时语音和数天 A100，且它学的是"特征->波形"的
    通用映射，和你转成谁无关，重复造轮子没有学习价值。核心算法（特征抽取 + 匹配）
    我们自己写，那才是 VC 的本体。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from .features import WavLMFeatureExtractor
from .matching import knn_convert
from .pitch import (analyze, apply_source_contour, log_f0_stats, shift_semitones,
                    speaker_f0_stats, synthesize as world_synthesize)


def _sha1(path: str) -> str:
    """源音频的内容指纹。文件名可以重复、可以被改动，指纹不会骗人。"""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def load_vocoder(device: str, prematched: bool = True):
    """prematched 权重是在"kNN 平均之后"的特征上微调的。

    这是个典型的训练/推理分布失配问题：kNN 取均值会让特征比原始特征更平滑，
    如果声码器只见过原始特征，遇到平滑特征就会产生沙哑和杂音。用 prematched
    版本能明显提升清晰度——记住这个思路，你以后调任何两阶段系统都会用到。
    """
    hifigan, h = torch.hub.load(
        "bshall/knn-vc", "hifigan_wavlm", trust_repo=True,
        prematched=prematched, device=device, progress=True,
    )
    return hifigan.eval(), h


@torch.inference_mode()
def synthesize(hifigan, feats: torch.Tensor, device: str) -> torch.Tensor:
    return hifigan(feats.unsqueeze(0).to(device)).squeeze().cpu()


def postprocess_pitch(wav: np.ndarray, source_path: str, target_dir: str,
                      mode: str, extra_semitones: float) -> np.ndarray:
    """在波形上重写 F0。HiFi-GAN 没有音高输入端口且我们不重训它，
    所以只能在它输出之后，用 WORLD 把声源和声道拆开，单独改声源。

    代价是要多做一次 WORLD 的分析-合成往返，即使什么都不改也会有轻微音质
    损失。所以默认 mode=off 时这个函数根本不会被调用，别为了不需要的功能
    付出音质。想知道这个代价有多大，跑 explore/03_pitch.py 听 shift 0 半音。

    contour  : 源的语调轮廓 + 目标的音域。针对深层特征音高失控的问题。
    source   : 完全保留你自己的音高，只换音色，不做性别迁移时用。
    roundtrip: F0 一个字节都不改，只走一遍 WORLD。这是诊断用的对照组 ——
               它和 off 的差距就是往返本身的纯代价，和 contour 的差距才是
               改动 F0 的代价。不分开测，你就不知道该怪谁。
    """
    src_wav, _ = librosa.load(source_path, sr=16000, mono=True)
    f0_out, sp, ap = analyze(wav)
    f0_src, _, _ = analyze(src_wav)

    if mode == "contour":
        src_stats = log_f0_stats(f0_src)
        tgt_stats = speaker_f0_stats(target_dir)
        f0_new = apply_source_contour(f0_out, f0_src, src_stats, tgt_stats)
    elif mode == "source":
        stats = log_f0_stats(f0_src)
        f0_new = apply_source_contour(f0_out, f0_src, stats, stats)
    else:
        f0_new = f0_out

    if extra_semitones:
        f0_new = shift_semitones(f0_new, extra_semitones)

    for name, f in (("处理前", f0_out), ("处理后", f0_new)):
        mu, sigma = log_f0_stats(f)
        if np.isfinite(mu):
            print(f"[pitch] {name} 中位音高 {np.exp(mu):6.1f}Hz  log标准差 {sigma:.3f}")

    return world_synthesize(f0_new, sp, ap)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="要被转换的音频（你的声音）")
    parser.add_argument("--target-dir", required=True, help="目标角色音频所在目录")
    parser.add_argument("--out", default="outputs/converted.wav")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--layer", type=int, default=6,
                        help="改这个必须换配套声码器，仅用于做层消融实验")
    parser.add_argument("--raw-vocoder", action="store_true",
                        help="用非 prematched 声码器，听听分布失配有多难听")
    parser.add_argument("--pitch", choices=["off", "contour", "source", "roundtrip"],
                        default="off",
                        help="contour=源语调+目标音域; source=保留自己的音高; "
                             "roundtrip=只走一遍 WORLD 不改 F0，用于隔离往返本身的代价")
    parser.add_argument("--pitch-shift", type=float, default=0.0,
                        help="额外平移多少个半音，12 为一个八度")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    extractor = WavLMFeatureExtractor(layer=args.layer, device=device)
    matching_set = extractor.build_matching_set(args.target_dir)

    src_feats = extractor.from_file(args.source)
    print(f"[source] {src_feats.shape[0]} 帧 ({src_feats.shape[0] / 50:.1f}s)")

    converted = knn_convert(src_feats, matching_set, topk=args.topk, device=device)

    hifigan, h = load_vocoder(device, prematched=not args.raw_vocoder)
    wav = synthesize(hifigan, converted, device).numpy()

    if args.pitch != "off" or args.pitch_shift:
        wav = postprocess_pitch(wav, args.source, args.target_dir,
                                args.pitch, args.pitch_shift)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 采样率从声码器配置里读，不要硬编码：换了声码器而采样率没跟着改，
    # 结果是音高和语速整体错位，且不会有任何报错。
    sf.write(out_path, wav, h.sampling_rate)

    # 每个输出旁边留一份参数快照。没有它，几十个实验文件跑完之后
    # 你将无法回答"这个 wav 到底是用哪个源、哪个池子生成的"。
    meta = {
        "source": str(Path(args.source).resolve()),
        "source_sha1": _sha1(args.source),
        "target_dir": str(Path(args.target_dir).resolve()),
        "matching_set_frames": int(matching_set.shape[0]),
        "layer": args.layer,
        "topk": args.topk,
        "prematched_vocoder": not args.raw_vocoder,
        "pitch_mode": args.pitch,
        "pitch_shift": args.pitch_shift,
        "sampling_rate": int(h.sampling_rate),
        "output_samples": int(wav.size),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    out_path.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out_path.resolve()}")
    print(f"[meta] {out_path.with_suffix('.json').name}  "
          f"源指纹 {meta['source_sha1']}  池 {meta['matching_set_frames']} 帧")


if __name__ == "__main__":
    main()
