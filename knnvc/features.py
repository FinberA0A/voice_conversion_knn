"""WavLM 帧级特征抽取。

为什么是 WavLM 而不是梅尔谱？
    梅尔谱里内容和音色是纠缠的——同一个音素，你说和角色说，梅尔谱长得很不一样。
    WavLM 这类自监督模型经过大规模无标注语音的掩码预测训练，中间层表示主要由
    "发的是什么音"决定，说话人信息被大幅削弱。这正是我们做检索匹配需要的性质。

为什么固定取第 6 层？
    自监督模型的层有明显分工：浅层偏声学细节（含大量说话人信息），
    深层偏语义/上下文，中间层音素信息最纯。knn-vc 原论文做了逐层实验，
    第 6 层在"内容保持"和"音色剥离"之间最平衡，其配套的 HiFi-GAN 声码器
    也是在第 6 层特征上训练的——层号必须和声码器一致，改了就得重训声码器。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

SR = 16000
DEFAULT_LAYER = 6


class WavLMFeatureExtractor:
    def __init__(self, layer: int = DEFAULT_LAYER, device: str = "cuda") -> None:
        self.layer = layer
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load(
            "bshall/knn-vc", "wavlm_large", trust_repo=True,
            device=str(self.device), progress=True,
        ).eval()

    @torch.inference_mode()
    def from_array(self, wav: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(wav).float().to(self.device).unsqueeze(0)
        # WavLM-Large 训练时对输入波形做了实例归一化，推理必须保持一致
        if getattr(self.model, "cfg", None) is not None and self.model.cfg.normalize:
            x = F.layer_norm(x, x.shape)
        feats, _ = self.model.extract_features(x, output_layer=self.layer,
                                               ret_layer_results=False)
        return feats.squeeze(0)  # [T, 1024]，帧率 50Hz

    def from_file(self, path: str | Path) -> torch.Tensor:
        wav, _ = librosa.load(str(path), sr=SR, mono=True)
        return self.from_array(wav)

    def build_matching_set(self, audio_dir: str | Path,
                           exts=(".wav", ".flac", ".mp3"),
                           cache_dir: str | Path = "outputs/.cache") -> torch.Tensor:
        """把目标角色的所有音频抽成一个大特征池，这就是 kNN 的检索库。

        池子越大覆盖的音素越全，转换质量越好；论文里 5~10 分钟接近饱和，
        再往上收益递减。低于 5 分钟则是明确的瓶颈——没有任何超参数能弥补
        池子里根本不存在的音素。

        结果带缓存：几十分钟音频过一遍 WavLM 要好几分钟，而做消融实验时
        池子通常不变，每次重抽纯属浪费。缓存键含层号和文件清单指纹，
        换层或改动目录会自动失效，不会拿到陈旧的池子。
        """
        files = sorted(p for p in Path(audio_dir).rglob("*") if p.suffix.lower() in exts)
        if not files:
            raise FileNotFoundError(f"{audio_dir} 下没有找到音频")

        sig = hashlib.sha1(
            (f"layer{self.layer}|"
             + "|".join(f"{p.name}:{p.stat().st_size}" for p in files)).encode()
        ).hexdigest()[:16]
        cache = Path(cache_dir) / f"pool_{Path(audio_dir).name}_L{self.layer}_{sig}.pt"

        if cache.exists():
            matching_set = torch.load(cache)
            print(f"[matching set] 命中缓存 {cache.name} -> {matching_set.shape[0]} 帧")
            return matching_set

        pool, total_sec = [], 0.0
        for p in tqdm(files, desc=f"抽取特征池 (layer {self.layer})", unit="file"):
            wav, _ = librosa.load(str(p), sr=SR, mono=True)
            total_sec += len(wav) / SR
            pool.append(self.from_array(wav).cpu())

        matching_set = torch.cat(pool, dim=0)
        print(f"[matching set] {len(files)} 个文件 / {total_sec / 60:.1f} 分钟 "
              f"-> {matching_set.shape[0]} 帧 x {matching_set.shape[1]} 维")
        if total_sec < 300:
            print("  警告: 不足 5 分钟，大量音素找不到合适的近邻。"
                  "先扩数据再调参，否则调什么都是徒劳")

        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(matching_set, cache)
        return matching_set
