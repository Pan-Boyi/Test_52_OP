# opbench —— 通用 1952 单算子上板验证与性能剖析

由两套内部单算子上板 harness 归纳而来：取其各自的长处，并把**算子相关的部分全部外移
到一个算子描述 JSON**，使流程脚本本身对算子无感知。

## 三个文件

| 文件 | 换算子要改吗 |
|---|---|
| `opbench.py` | 不改。host 侧驱动，七个步骤 |
| `opbench_device.py` | 不改。板上执行，ctypes 直调 `libascendcl.so` |
| `opbench_config.json` | 不改（除了板子地址）。机器与板子配置 |
| `ops/<算子>.json` | **只改这个。** 输入/输出/属性/填充规则 |

## 七个步骤

```
case    按算子描述生成输入 .bin 和 manifest.json
om      写 singleop.json，调 atc 出 .om
push    组 bundle（model/ + case/ + 设备侧脚本 + SHA256SUMS）传到板子
run     在板上执行，可选套 msprof
pull    把 PROF 目录和设备输出取回本地
export  用 msprof.py export summary 把原始 PROF 解成 CSV
parse   读 op_summary_*.csv，打各 pipe 的性能表
```

每步可单独跑，也可连起来。

## 用法

```bash
# 一条龙
python3 opbench.py ops/dec_enc_fused_kernel.json

# 只在本地出数据和 .om（不碰板子，不需要配凭据）
python3 opbench.py ops/dec_enc_fused_kernel.json --steps case,om

# 只跑一次、不采 profiling（先确认能跑通再谈性能）
python3 opbench.py ops/dec_enc_fused_kernel.json --no-msprof --repeat 1

# 已经有 PROF 目录了，只重新解析
python3 opbench.py ops/dec_enc_fused_kernel.json --steps parse
```

板子密码用环境变量传，别写进配置文件，也别让它进 shell 历史：

```bash
read -rs OPBENCH_REMOTE_PASSWORD && export OPBENCH_REMOTE_PASSWORD
```

## 算子描述

```json
{
  "name": "dec_enc_fused_kernel",
  "op_type": "DecEncFusedKernel",
  "seed": 0,
  "inputs": [
    {"name": "x", "shape": [1,4,2160,3840], "dtype": "float32", "format": "ND",
     "fill": {"kind": "uniform", "lo": 0.0, "hi": 1.0}},
    {"name": "some_optional", "absent": true}
  ],
  "outputs": [
    {"name": "y", "shape": [1,32,2160,3840], "dtype": "float32", "format": "ND"}
  ],
  "attrs": [
    {"name": "mode", "type": "int", "value": 3},
    {"name": "tile", "type": "list_int", "value": [6, 960]}
  ]
}
```

`inputs` 的顺序**就是算子的 IR ABI 顺序**，不能跳过槽位 —— 不用的 optional 输入写
`{"name": "...", "absent": true}`，脚本会在 `singleop.json` 里补
`RESERVED`/`UNDEFINED` 占位，设备侧补空 desc。

`fill.kind` 可选：`uniform{lo,hi}`、`normal{mean,std}`、`const{value}`、
`arange{start,step}`、`randint{lo,hi}`、`zeros`、`ones`。
每个输入的随机种子由 `(seed, 输入名)` 决定 —— 改一个输入的填充规则不会把别的输入的
数据也变掉，便于单独排查某一路。

`attrs` 的 `type` 支持 `int` / `bool` / `float` / `string` / `list_int` / `list_float`。

dtype 支持 `float32` `float16` `bfloat16` `int8` `int32` `uint8` `int16` `uint16`
`uint32` `int64` `uint64` `double` `bool`。

## 会让人白花一轮的八条

下面每一条都是实际踩过并记录下来的，不是风格偏好。

1. **`profile.msprof` 必须给绝对路径，不能搜 PATH。** 远端那个 ssh 是非交互 shell，
   读不到 `~/.bashrc` 里 source 的 `set_env.sh`，于是明明装了也报「PATH 里没有」；
   或者机器上有好几套 CANN，搜到的不是这次要用的那一套 —— 数据照出，来源不明。

2. **PROF 目录靠「跑之前 / 跑之后」的集合差认，不靠时间戳也不靠名字猜。** 差集为空
   = msprof 没产出（退出 2），多于一个 = 并发或残留（退出 3），两种都当失败。

3. **报最小值，不报平均。** host 抖动和别的进程抢核只会让某几次变慢，不会让它变快。
   平均和最大一并给出，用来判断抖动有多大。

4. **`singleop.json` 的属性必须一个不少地全写上。** ACL 是按属性的**值**匹配 `.om`
   的，多一个少一个都匹配不上，而报出来是「算子没找到」（100024）。

5. **缺席的 optional 输入不能省略**，要占位，否则后面的输入全部错位到前一个 IR 槽上。

6. **`remote.auth` 写 `"auto"` 是个陷阱。** 有些板子禁掉了公钥登录只认密码，而本机
   没装 `sshpass` 时 `auto` 会不声不响退回密钥 —— 表现是一句和认证毫无关系的
   「远端建目录失败（rc=255）」。确定只能用密码时就写死 `"password"`。

7. **远端目录每次重建（`rm -rf`）。** 上一轮残留的 `.om` 会被 `aclopSetModelDir`
   一起加载，然后按属性值匹配到旧的那个上去，静默跑出上一版的数。
   因此 `remote.dir` 有护栏：必须是绝对路径，且不能是 `/` `/root` `/home` `/tmp`
   这类目录 —— 配错一个字就会在板子上删掉别人的东西。

8. **先探 ssh，再干本地重活。** 进不去的话没必要先花几十秒编 `.om`。

另外两条来自设备侧：

- **算子名必须和装进 OPP 的那个逐字一致**，差一个下划线也是 100024。
- **`warmup` 至少给 1**。第一次下发连带算子选择、模型匹配和首次 kernel 加载，比后面
  每次都慢得多；采 profiling 时不预热，第一行数据明显偏大而看不出原因。

## 各 pipe 列的含义

`op_summary_*.csv` 里取这七列：

| 列 | 对应 |
|---|---|
| `total_exe_time(us)` | 算子总耗时 |
| `mac_exe_time(us)` | **CUBE**（纯向量算子恒为 0） |
| `scalar_exe_time(us)` | SCALAR |
| `mte1_exe_time(us)` | MTE1（L1→L0） |
| `mte2_exe_time(us)` | MTE2（GM→L1/UB） |
| `mte3_exe_time(us)` | MTE3（UB→GM） |
| `fixpipe_time(us)` | FIXPIPE（纯向量算子恒为 0） |

举个例子：它是写出为主的带宽瓶颈算子，所以预期 `mte3` 压倒性
占比、`mte2` 约为其八分之一、`mac` 和 `fixpipe` 恒为 0。实测偏离这个形状就说明
有问题 —— 比如 `mac` 非 0 意味着有 cube 指令被编进去了。

## 已验证到哪一步

本地路径都真跑过：配置校验、`rm -rf` 护栏（正负两向）、算子描述解析、
`singleop.json` 生成（含占位）、`case` 步的数据正确性（`arange` / `const` /
`randint` 逐值核对）、`parse` 步（含 BOM、滤掉非目标算子、各 pipe 统计）。

**`om` / `push` / `run` / `pull` / `export` 五步需要 atc 和板子，尚未实跑。**
它们的命令行每个 flag 都核对过来源，但「写对了」和「跑通了」是两件事。
