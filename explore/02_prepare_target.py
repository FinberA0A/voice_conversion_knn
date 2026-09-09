"""把已经分离好的长 wav 切成 5~10 秒片段，写入目标特征池目录。

用法:
    python explore/02_prepare_target.py path/to/vocals.wav --out data/target/角色名
    python explore/02_prepare_target.py path/to/wav目录 --out data/target/角色名

不处理 mkv/mp4。输入必须是 wav（UVR 等人声提取后的结果）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 16000
MIN_SEC = 3.0
MAX_SEC = 10.0
TARGET_SEC = 8.0


def collect_wavs(src: Path) -> list[Path]:
    if src.is_file():
        if src.suffix.lower() != ".wav":
            raise SystemExit(f"只接受 wav，收到的是 {src.suffix}")
        return [src]
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() == ".wav")
        if not files:
            raise SystemExit(f"{src} 下没有 wav")
        return files
    raise SystemExit(f"找不到: {src}")


def load_wav(path: Path) -> np.ndarray:
    wav, _ = librosa.load(str(path), sr=SR, mono=True)
    if wav.size < int(SR * MIN_SEC):
        raise SystemExit(f"{path.name} 太短（{wav.size / SR:.1f}s），至少要几秒钟。")
    return wav


def split_chunks(wav: np.ndarray) -> list[np.ndarray]:
    """按约 8 秒切开。UVR 人声整体偏安静，按能量检测静音会把大部分内容丢掉。"""
    step = int(SR * TARGET_SEC)
    min_n = int(SR * MIN_SEC)
    if wav.size <= int(SR * MAX_SEC):
        return [wav]
    chunks = [wav[i : i + step] for i in range(0, wav.size, step)]
    if chunks[-1].size < min_n:
        chunks.pop()
    return chunks if chunks else [wav]


def next_index(out_dir: Path) -> int:
    existing = [int(p.stem) for p in out_dir.glob("*.wav") if p.stem.isdigit()]
    return max(existing, default=0) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="一条长 wav，或装满 wav 的目录")
    parser.add_argument("--out", default="data/target/character", help="切片输出目录")
    args = parser.parse_args()

    wavs = collect_wavs(Path(args.input))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = next_index(out_dir)
    n_written = 0
    total = 0.0

    for src in wavs:
        wav = load_wav(src)
        chunks = split_chunks(wav)
        kept = sum(c.size for c in chunks) / SR
        print(f"[load] {src.name}  {wav.size / SR:.1f}s  -> {len(chunks)} 条 / {kept:.1f}s")
        for chunk in chunks:
            path = out_dir / f"{idx:04d}.wav"
            sf.write(path, chunk, SR)
            print(f"  {path.name}  {chunk.size / SR:.1f}s")
            idx += 1
            n_written += 1
            total += chunk.size / SR

    print(f"[done] {n_written} 条，合计 {total:.1f}s  ->  {out_dir.resolve()}")
    if total < 300:
        print("  提示: 特征池建议 5~10 分钟。不够的话继续往同一目录追加即可。")


if __name__ == "__main__":
    main()
