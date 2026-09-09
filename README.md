# 声音转换入门 · 手写 knn-vc

一个用于学习语音转换（Voice Conversion）原理的最小实现。核心算法零训练、零参数，
两百行代码内跑通「我的声音 → 目标角色的声音」。

仅供非商业的个人实验。声音属于人格权益，请只使用你自己的声音或已获授权的素材，
产出内容注明为 AI 合成。

## 原理

```
源音频 ──WavLM(第6层)──> 帧级特征 ──kNN 检索替换──> 目标角色特征 ──HiFi-GAN──> 波形
                                        ↑
                          目标角色音频抽成的特征池（5~10 分钟）
```

关键在于 WavLM 中间层特征的一个性质：**帧之间的距离主要由「发的是什么音」决定，
而不是「谁在发音」**。所以拿源音频某一帧去目标角色的特征池里检索，找回来的就是
角色发同一个音时的帧。逐帧替换后，内容轨迹保留，音色整体换成了角色的。

没有编码器-解码器，没有对抗训练，没有说话人嵌入。这是理解 VC 最好的起点。

## 环境

RTX 50 系（Blackwell, sm_120）必须用 cu128 及以上的 PyTorch，这是最容易踩的坑。

```powershell
conda create -n vc python=3.10 -y
conda activate vc
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

验证（应输出 `True (12, 0)`，第二项是算力号）：

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"
```

## 数据

```
data/
├── source/            # 你自己的录音，16k 以上，安静环境，3~15 秒一条
└── target/<角色名>/    # 目标角色语音，总长 5~10 分钟，切成 5~15 秒片段
```

动漫/游戏角色素材通常是录音棚干音，这是它相比翻录人声的最大优势。但要注意剔除：
带 BGM 的、有战斗音效的、混响特效很重的、以及喊叫或哭腔等极端发声——这些会污染
特征池，检索时被匹配到就会产生怪声。

## 三步走

```powershell
# 1. 先把声学理论跑一遍，看图理解 STFT / 梅尔谱 / 共振峰 / F0 / 相位丢失
python explore/01_acoustics.py data/target/角色名/0001.wav

# 2. 端到端转换
python -m knnvc.convert --source data/source/me.wav --target-dir data/target/角色名

# 3. 客观评估（见 eval/）
```

首次运行会从 GitHub 下载 WavLM-Large（约 1.2GB）和 HiFi-GAN 权重，缓存在
`~/.cache/torch/hub`。网络不通的话搜索关键词 `torch hub 镜像` 或手动下权重。

## 你该做的消融实验

理解一个方法最快的路径是把它拆坏。按顺序试，每次只改一个变量，用耳朵和评估指标一起判断：

| 变量 | 试什么 | 你应该观察到 |
|---|---|---|
| `--topk` | 1 / 4 / 20 / 100 | 1 有颗粒噪声，过大则发音变糊、音色被抹向"平均人声" |
| 特征池时长 | 30 秒 / 2 分钟 / 10 分钟 | 池子小时生僻音素检索不到，输出含糊；`[knn]` 打印的距离会明显偏大 |
| `--layer` | 1 / 6 / 12 | 浅层残留源说话人音色，深层内容对了但声码器无法合成（层与声码器绑定） |
| `--raw-vocoder` | 加上此参数 | 沙哑、杂音 —— 这就是训练/推理分布失配的听觉表现 |
| 源音频语言 | 中文源 → 日文角色池 | 跨语言时目标缺少对应音素，是这个方法的固有短板 |

## 已知局限

- **音高不可控**。这个方法不显式建模 F0，输出音高由检索到的目标帧决定：
  既不完全跟随源，也不精确等于目标，跨性别转换时会飘。想要可控就必须
  引入显式的 F0 条件建模，那已经是 RVC / so-vits-svc 的范畴了。
- **需要目标角色 5 分钟以上素材**，做不到几秒钟的 zero-shot。
- **跨语言效果差**，源和目标最好是同一语言。

## 延伸关键词

`kNN-VC`、`WavLM`、`ContentVec`、`HiFi-GAN`、`neural source filter`、
`RVC retrieval based voice conversion`、`VITS normalizing flow`、`seed-vc`
