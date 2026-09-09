"""模型解剖：把 WavLM 和 HiFi-GAN 拆开，看清楚数据流过时形状怎么变。

用法:
    python explore/02_model_anatomy.py

不要只读输出数字，对照着看三件事：
    1. WavLM 的 CNN 部分如何把 16000Hz 的波形压成 50Hz 的帧序列（320 倍下采样）
    2. HiFi-GAN 如何把 50Hz 的帧序列还原成 16000Hz 波形（320 倍上采样）
    3. 参数量集中在哪里 —— 你会发现 WavLM 比声码器大一个数量级
"""

from __future__ import annotations

import numpy as np
import torch


def human(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.1f}K"


def params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def anatomy_wavlm(device: str) -> None:
    print("=" * 72)
    print("WavLM-Large：波形 -> 帧级特征")
    print("=" * 72)

    model = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True,
                           device=device, progress=True).eval()

    # --- 第一段：CNN 特征编码器 ---
    # 7 层一维卷积，纯粹做时间轴上的降采样。它没有任何注意力机制，
    # 只看局部波形，作用相当于一个"可学习的、替代 STFT 的前端"。
    conv_layers = model.feature_extractor.conv_layers
    print("\n[1] CNN feature encoder —— 相当于可学习版的 STFT 前端")
    print(f"{'layer':>6} {'kernel':>7} {'stride':>7} {'累积下采样':>11} {'感受野(样本)':>13} {'感受野(ms)':>11}")

    cumulative_stride, receptive = 1, 1
    for i, layer in enumerate(conv_layers):
        conv = layer[0]
        k, s = conv.kernel_size[0], conv.stride[0]
        # 感受野递推：新感受野 = 旧感受野 + (kernel-1) * 之前所有stride的乘积
        receptive = receptive + (k - 1) * cumulative_stride
        cumulative_stride *= s
        print(f"{i:>6} {k:>7} {s:>7} {cumulative_stride:>11} {receptive:>13} "
              f"{receptive / 16000 * 1000:>10.1f}")

    print(f"\n  总下采样率 = {cumulative_stride}  ->  16000Hz 波形变成 "
          f"{16000 / cumulative_stride:.0f}Hz 的帧序列")
    print(f"  即每帧代表 {cumulative_stride / 16000 * 1000:.0f}ms，"
          f"每帧能看到 {receptive / 16000 * 1000:.0f}ms 的波形")
    print("  ^ 这就是 explore/01 里 HOP=320 的由来，我们特意和它对齐")

    # --- 第二段：Transformer ---
    encoder_layers = model.encoder.layers
    first = encoder_layers[0]
    dim = first.self_attn.embed_dim
    heads = first.self_attn.num_heads
    print(f"\n[2] Transformer encoder —— {len(encoder_layers)} 层, "
          f"隐藏维度 {dim}, {heads} 个注意力头 (每头 {dim // heads} 维)")
    print("  CNN 只看局部，Transformer 让每一帧和全句所有帧交互，")
    print("  于是特征里编码的不再是'这 25ms 的波形长什么样'，而是")
    print("  '在整句话的上下文里，此刻在发哪个音'。这正是 kNN 检索需要的。")

    # --- 参数量分布 ---
    print(f"\n[3] 参数量")
    print(f"  CNN 前端      {human(params(model.feature_extractor)):>10}")
    print(f"  Transformer   {human(params(model.encoder)):>10}")
    print(f"  合计          {human(params(model)):>10}")
    print("  ^ 绝大部分参数在 Transformer，语音理解的重活是它干的")

    # --- 实测形状 ---
    dummy = torch.randn(1, 16000, device=device)  # 1 秒静噪
    with torch.inference_mode():
        if getattr(model, "cfg", None) is not None and model.cfg.normalize:
            dummy = torch.nn.functional.layer_norm(dummy, dummy.shape)
        feats, _ = model.extract_features(dummy, output_layer=6,
                                          ret_layer_results=False)
    print(f"\n[4] 实测: 1 秒波形 [1, 16000] -> 第6层特征 {list(feats.shape)}")

    # 帧数比 16000/320=50 少 1，因为这些卷积都没有 padding：
    # 每层按 floor((L-k)/s)+1 计算，末尾凑不满一个卷积核的采样点被直接丢弃。
    length = 16000
    trace = [length]
    for layer in conv_layers:
        conv = layer[0]
        length = (length - conv.kernel_size[0]) // conv.stride[0] + 1
        trace.append(length)
    print(f"  逐层精确长度: {' -> '.join(map(str, trace))}")
    print(f"  理论 16000/{cumulative_stride}={16000 // cumulative_stride} 帧，"
          f"实际 {length} 帧，差的这一帧是无 padding 卷积在末尾的截断损失")
    print(f"  后果: 声码器还原出 {length * cumulative_stride} 点，比输入短 "
          f"{16000 - length * cumulative_stride} 点({(16000 - length * cumulative_stride) / 16} ms)")

    del model
    torch.cuda.empty_cache()
    return feats.shape[1]


def anatomy_hifigan(device: str, n_frames: int) -> None:
    print("\n" + "=" * 72)
    print("HiFi-GAN：帧级特征 -> 波形")
    print("=" * 72)

    hifigan, h = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True,
                                prematched=True, device=device, progress=True)
    hifigan = hifigan.eval()

    rates = list(h.upsample_rates)
    total = int(np.prod(rates))
    print(f"\n[1] 上采样倍率 {rates}  连乘 = {total}")
    print(f"  配置里的采样率 h.sampling_rate = {h.sampling_rate}")
    print(f"  {'一致' if total == 320 else '注意：与 WavLM 的 320 不一致！'}"
          " —— 必须严格等于 WavLM 的下采样率，否则时长和音高都会错")

    # 上采样不会让信号变长（时长恒定），它提高的是时间分辨率。
    # 所以中间层的长度必须配上它自己的"等效采样率"才有物理意义。
    print(f"\n[2] 逐级上采样：时长恒定不变，变的是时间分辨率")
    length, remaining = n_frames, total
    for i, r in enumerate(rates):
        length *= r
        remaining //= r
        eq_sr = h.sampling_rate / remaining
        print(f"  第{i}级 x{r:<3} -> 长度 {length:>7}  等效采样率 {eq_sr:>7.0f}Hz  "
              f"时长 {length / eq_sr * 1000:>7.1f}ms")

    print(f"\n[3] MRF (Multi-Receptive Field) 残差块")
    print(f"  resblock kernel sizes    {list(h.resblock_kernel_sizes)}")
    print(f"  resblock dilations       {list(h.resblock_dilation_sizes)}")
    print("  每级上采样后，用 3 个不同卷积核+不同空洞率的残差块并联再求平均。")
    print("  目的是同时captures短时细节(小核)和长时结构(大核+空洞)，")
    print("  语音里既有几毫秒的爆破音，又有上百毫秒的元音共振，单一尺度顾不过来。")

    print(f"\n[4] 参数量  生成器 {human(params(hifigan))}")
    print("  ^ 对比 WavLM 的 300M+，声码器小得多。")
    print("    它只干'特征->波形'这一件事，和转换成谁无关，所以可以复用别人的权重。")

    with torch.inference_mode():
        dummy = torch.randn(1, n_frames, 1024, device=device)
        wav = hifigan(dummy)
    print(f"\n[5] 实测: {n_frames} 帧特征 [1, {n_frames}, 1024] -> 波形 {list(wav.shape)}")
    print(f"  长度比 = {wav.shape[-1] / n_frames:.0f}，与上采样连乘 {total} 一致")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}\n")
    n_frames = anatomy_wavlm(device)
    anatomy_hifigan(device, n_frames)
    print("\n" + "=" * 72)
    print("记住这条链路的形状变化：")
    print("  [16000 采样点] --WavLM CNN /320--> [50 帧, 512] --Transformer--> [50 帧, 1024]")
    print("  [50 帧, 1024] --kNN 替换(无参数)--> [50 帧, 1024] --HiFiGAN x320--> [16000 采样点]")
    print("=" * 72)


if __name__ == "__main__":
    main()
