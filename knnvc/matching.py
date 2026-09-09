"""kNN 回归：整个 knn-vc 的全部"模型"就在这个文件里，没有任何可训练参数。

思路一句话说清：
    源音频第 t 帧的 WavLM 特征代表"此刻在发某个音"。到目标角色的特征池里
    找出最像的 k 帧——它们必然是角色在发同一个音的时刻——取平均替换掉原特征。
    于是内容轨迹保留了，而特征本身完全来自目标角色。

为什么取平均而不是取最近的 1 个？
    单个近邻会让相邻帧在特征空间里来回跳变，合成出来有明显的颗粒感和杂音。
    平均 k 个起到平滑作用。k 太大则音色被抹平、发音变糊，实践中 4 是个好起点。
"""

from __future__ import annotations

import torch


def cosine_distance(query: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
    """[Tq, D] x [Tp, D] -> [Tq, Tp]

    用余弦而非欧氏距离：WavLM 特征的模长更多反映能量/响度，方向才编码音素身份。
    """
    q = torch.nn.functional.normalize(query, dim=-1)
    p = torch.nn.functional.normalize(pool, dim=-1)
    return 1.0 - q @ p.T


def knn_convert(
    query: torch.Tensor,
    matching_set: torch.Tensor,
    topk: int = 4,
    chunk: int = 512,
    device: str = "cuda",
    report: bool = True,
) -> torch.Tensor:
    """把源特征逐帧替换为目标特征池中 topk 近邻的均值。

    分块处理是因为距离矩阵是 [Tq, Tp]：10 分钟的池子有 3 万帧，
    源音频 30 秒有 1500 帧，全量矩阵约 180MB，再长就会撑爆 8GB 显存。
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    pool = matching_set.to(dev)
    query = query.to(dev)
    #以上是为了将数据移动到GPU上进行计算

    out, nn_dists = [], []
    for i in range(0, query.shape[0], chunk):
        block = query[i: i + chunk]
        dists = cosine_distance(block, pool)
        best = dists.topk(k=topk, largest=False, dim=-1)
        out.append(pool[best.indices].mean(dim=1))#indices是帧号
        nn_dists.append(best.values[:, 0])

    converted = torch.cat(out, dim=0)

    if report:
        d = torch.cat(nn_dists)
        print(f"[knn] top1 余弦距离 中位数 {d.median():.4f} / 90分位 {d.quantile(0.9):.4f}")
        print("      距离偏大说明目标池里缺少对应音素，换更长或内容更丰富的素材")
    return converted.cpu()
