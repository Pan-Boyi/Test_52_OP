#!/usr/bin/env python3
"""按 DecEncFusedKernel 的形状族生成 opbench 用例 JSON。

算子支持 540x960 的倍数族：H = 540 * hBlk、W = 960 * wBlk，hBlk/wBlk 都取 1..4，
tile 参数的首维 nTiles = hBlk * wBlk。dtype 是 fp32 或 fp16，**十个张量同 dtype**。

手写这些 JSON 很容易把 nTiles 和形状写得不自洽（给短了算子会读到参数缓冲里的垃圾，
给长了 tiling 直接拒），所以用这个脚本生成 —— nTiles 由形状算出，不可能对不上。

    python3 gen_dec_enc_fused_kernel.py              # 生成默认那三个用例
    python3 gen_dec_enc_fused_kernel.py 1620 2880 float16   # 生成任意一个

填充范围的约定（pow 路的定义域要求）：
  x        [0, 1]      —— exp(s*ln x) 不接受负底数
  指数     (0, 3]      —— x == 0 且 s <= 0 时算子一律出 0，不要去测那个角
  scale    [0.5, 2]
  bias/mix [-1, 1]
"""

import json
import os
import sys

BLK_H, BLK_W = 540, 960
CIN, COUT = 4, 32


def spec(h: int, w: int, dtype: str) -> dict:
    if h % BLK_H or w % BLK_W or not (1 <= h // BLK_H <= 4) or not (1 <= w // BLK_W <= 4):
        raise SystemExit(f"{h}x{w} 不在支持的形状族里：H 取 540 的 1..4 倍、W 取 960 的 1..4 倍")
    if dtype not in ("float32", "float16"):
        raise SystemExit(f"dtype 只支持 float32 / float16，给的是 {dtype}")
    n = (h // BLK_H) * (w // BLK_W)

    def t(name, shape, lo, hi):
        return {"name": name, "shape": shape, "dtype": dtype, "format": "ND",
                "fill": {"kind": "uniform", "lo": lo, "hi": hi}}

    return {
        "name": f"dec_enc_fused_kernel_{h}x{w}_{dtype}",
        "op_type": "DecEncFusedKernel",
        "seed": 0,
        "_comment": f"H={h} W={w} hBlk={h // BLK_H} wBlk={w // BLK_W} nTiles={n} dtype={dtype}",
        "inputs": [
            t("x", [1, CIN, h, w], 0.0, 1.0),
            t("channel_scale", [1, CIN, 1, 1], 0.5, 2.0),
            t("global_exponent", [1, 1, 1, 1], 0.5, 3.0),
            t("global_bias", [1, 1, 1, 1], -1.0, 1.0),
            t("channel_mix", [1, CIN, CIN, 1, 1], -1.0, 1.0),
            t("tile_scale", [n, CIN, 1, 1], 0.5, 2.0),
            t("tile_exponent", [n, 1, 1, 1], 0.5, 3.0),
            t("tile_bias", [n, 1, 1, 1], -1.0, 1.0),
            t("tile_mix", [n, CIN, CIN, 1, 1], -1.0, 1.0),
        ],
        "outputs": [{"name": "y", "shape": [1, COUT, h, w], "dtype": dtype, "format": "ND"}],
        "attrs": [],
    }


def write(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"  {os.path.basename(path):42s} {data['_comment']}")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) == 4:
        h, w, dt = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
        write(os.path.join(here, f"dec_enc_fused_kernel_{h}x{w}_{dt}.json"), spec(h, w, dt))
        return

    # 默认三个用例，各有各的用处：
    print("生成：")
    # 1) 基线 —— 和之前上板跑过的那组完全一样，便于对比性能数据
    write(os.path.join(here, "dec_enc_fused_kernel.json"), spec(2160, 3840, "float32"))
    # 2) 同形状的 fp16 —— 单独验 dtype 这条轴
    write(os.path.join(here, "dec_enc_fused_kernel_fp16.json"), spec(2160, 3840, "float16"))
    # 3) **最该先跑的一个**：W < 3840，于是块号算式里的进制 wBlk != 4。
    #    块号写死成常量 4 的实现只有 W == 3840 时是对的，这个用例是唯一能在板上
    #    抓住它的 —— 而且抓法是出错数，不是报错。
    write(os.path.join(here, "dec_enc_fused_kernel_small.json"), spec(1080, 1920, "float32"))


if __name__ == "__main__":
    main()
