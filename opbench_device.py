#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""板上执行单算子 —— ctypes 直调 libascendcl.so，不依赖 pyACL，板上不需要编译器。

这个文件由 host 侧的 opbench.py 随 bundle 一起传到板子上，由 run 步骤调起。它只读
case/manifest.json，所以**对算子无感知**：输入顺序、dtype、形状、属性全部来自 manifest。

    python3 opbench_device.py --case-dir case --om model/xxx.om \
            --device 0 --repeat 10 --warmup 1

只用标准库 + ctypes：板子上通常没有 numpy，也不该为了跑一次算子去装。

================================================================================
三条会让人白查很久的 ACL 规则
================================================================================
1. **缺席的 optional 输入不能从数组里删掉。** 必须在原位置传 ACL_DT_UNDEFINED /
   ACL_FORMAT_UNDEFINED 的 desc 加 nullptr/0 的 DataBuffer。压缩掉会改变 numInputs
   和后面所有输入的位置，于是模型明明加载成功，执行时报 MatchOpModel 100024
   （「算子没找到」）—— 和真正原因毫无关系。

2. **aclopSetModelDir 只吃目录，而且会把目录下所有 .om 都加载进来。** 残留的旧 .om
   会按属性值被匹配上，静默跑出上一版的数。所以走这条回退路径时强制只许有一个 .om。
   能用 aclopLoad（直接吃 .om 的字节）就优先用它，没有这个歧义。

3. **算子名必须和装进 OPP 的那个逐字一致。** 差一个下划线（比如 MyOp5102 写成
   MyOp_5102）就是 100024，而报错不会告诉你是名字的问题。
"""

import argparse
import ctypes
import json
import os
import sys
import time

ACL_SUCCESS = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_FORMAT_ND = 2
# 缺席 optional 用的两个哨兵值，见文件头第 1 条。
ACL_DT_UNDEFINED = -1
ACL_FORMAT_UNDEFINED = -1

c_i = ctypes.c_int
c_i64 = ctypes.c_int64
c_vp = ctypes.c_void_p
c_cp = ctypes.c_char_p
c_sz = ctypes.c_size_t
c_f = ctypes.c_float
c_u8 = ctypes.c_uint8


def die(msg):
    print("\n[DEVICE X] %s" % msg, file=sys.stderr)
    sys.exit(1)


def check(rc, what):
    if rc != ACL_SUCCESS:
        die("%s 返回 %d" % (what, rc))


def load_acl():
    """绑定 libascendcl.so。签名逐个声明 —— 不声明的话 ctypes 默认 int 返回值，
    64 位指针会被截成 32 位，表现是一个随机的 null 检查失败。
    """
    for name in ("libascendcl.so", "libascendcl.so.1"):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            lib = None
    if lib is None:
        die("加载不到 libascendcl.so。\n"
            "  板上要么没装 CANN runtime，要么 LD_LIBRARY_PATH 里没有它 ——\n"
            "  非交互 ssh 读不到 ~/.bashrc，所以把 set_env.sh 配进 remote.cann_env。")

    sigs = [
        ("aclInit", [c_cp], c_i),
        ("aclFinalize", [], c_i),
        ("aclrtSetDevice", [c_i], c_i),
        ("aclrtResetDevice", [c_i], c_i),
        ("aclrtCreateStream", [ctypes.POINTER(c_vp)], c_i),
        ("aclrtDestroyStream", [c_vp], c_i),
        ("aclrtSynchronizeStream", [c_vp], c_i),
        ("aclrtMalloc", [ctypes.POINTER(c_vp), c_sz, c_i], c_i),
        ("aclrtFree", [c_vp], c_i),
        ("aclrtMemcpy", [c_vp, c_sz, c_vp, c_sz, c_i], c_i),
        ("aclCreateTensorDesc", [c_i, c_i, ctypes.POINTER(c_i64), c_i], c_vp),
        ("aclDestroyTensorDesc", [c_vp], None),
        ("aclCreateDataBuffer", [c_vp, c_sz], c_vp),
        ("aclDestroyDataBuffer", [c_vp], c_i),
        ("aclopCreateAttr", [], c_vp),
        ("aclopDestroyAttr", [c_vp], None),
        ("aclopSetAttrInt", [c_vp, c_cp, c_i64], c_i),
        ("aclopSetAttrBool", [c_vp, c_cp, c_u8], c_i),
        ("aclopSetAttrFloat", [c_vp, c_cp, c_f], c_i),
        ("aclopSetAttrString", [c_vp, c_cp, c_cp], c_i),
        ("aclopSetAttrListInt", [c_vp, c_cp, c_i, ctypes.POINTER(c_i64)], c_i),
        ("aclopSetAttrListFloat", [c_vp, c_cp, c_i, ctypes.POINTER(c_f)], c_i),
        ("aclopExecuteV2", [c_cp, c_i, ctypes.POINTER(c_vp), ctypes.POINTER(c_vp),
                            c_i, ctypes.POINTER(c_vp), ctypes.POINTER(c_vp),
                            c_vp, c_vp], c_i),
        ("aclopSetModelDir", [c_cp], c_i),
    ]
    for name, argtypes, restype in sigs:
        fn = getattr(lib, name, None)
        if fn is None:
            continue           # 可选符号（aclopLoad 等）缺了由调用点处理
        fn.argtypes = argtypes
        fn.restype = restype
    # aclopLoad 不是所有版本都导出，单独绑。
    fn = getattr(lib, "aclopLoad", None)
    if fn is not None:
        fn.argtypes = [c_vp, c_sz]
        fn.restype = c_i
    return lib


def load_model(acl, om_path):
    """优先 aclopLoad（吃字节），否则退回 aclopSetModelDir。见文件头第 2 条。

    aclopLoad 要求那块内存在整个执行期间都活着，所以 buffer 要被调用方持有 ——
    这里返回它，让 main 保存引用。提前释放会在执行时拿到已回收的模型。
    """
    with open(om_path, "rb") as f:
        blob = f.read()
    if not blob:
        die("om 是空文件：%s" % om_path)

    fn = getattr(acl, "aclopLoad", None)
    if fn is not None:
        keepalive = ctypes.create_string_buffer(blob, len(blob))
        check(fn(ctypes.cast(keepalive, c_vp), len(blob)), "aclopLoad")
        print("  模型：aclopLoad（%d 字节）" % len(blob))
        return keepalive

    fn = getattr(acl, "aclopSetModelDir", None)
    if fn is None:
        die("这套 CANN runtime 既没有 aclopLoad 也没有 aclopSetModelDir")
    d = os.path.dirname(os.path.abspath(om_path)) or "."
    sibs = [f for f in os.listdir(d) if f.endswith(".om")]
    if len(sibs) != 1:
        die("回退到 aclopSetModelDir 时 %s 下必须只有一个 .om，实际 %d 个：%s\n"
            "  见文件头第 2 条：它会把目录下所有 .om 一起加载。"
            % (d, len(sibs), ", ".join(sorted(sibs))))
    check(fn(d.encode()), "aclopSetModelDir")
    print("  模型：aclopSetModelDir(%s)" % d)
    return None


def build_attr(acl, attrs):
    """按 manifest 里的属性表建 aclopAttr。

    **属性必须一个不少地全设上，而且值要和 singleop.json 里的逐一相同** —— ACL 是按
    属性值匹配 .om 的，差一个就是 100024。manifest 里的表直接来自算子描述，和生成
    singleop.json 用的是同一份数据，所以这里不会和 .om 不一致。
    """
    attr = acl.aclopCreateAttr()
    if not attr:
        die("aclopCreateAttr 返回 null")
    for a in attrs:
        name = a["name"].encode()
        t = a["type"]
        v = a["value"]
        if t == "int":
            check(acl.aclopSetAttrInt(attr, name, int(v)), "aclopSetAttrInt(%s)" % a["name"])
        elif t == "bool":
            check(acl.aclopSetAttrBool(attr, name, 1 if v else 0),
                  "aclopSetAttrBool(%s)" % a["name"])
        elif t == "float":
            check(acl.aclopSetAttrFloat(attr, name, float(v)),
                  "aclopSetAttrFloat(%s)" % a["name"])
        elif t == "string":
            check(acl.aclopSetAttrString(attr, name, str(v).encode()),
                  "aclopSetAttrString(%s)" % a["name"])
        elif t == "list_int":
            arr = (c_i64 * len(v))(*[int(x) for x in v])
            check(acl.aclopSetAttrListInt(attr, name, len(v), arr),
                  "aclopSetAttrListInt(%s)" % a["name"])
        elif t == "list_float":
            arr = (c_f * len(v))(*[float(x) for x in v])
            check(acl.aclopSetAttrListFloat(attr, name, len(v), arr),
                  "aclopSetAttrListFloat(%s)" % a["name"])
        else:
            die("属性 %s 的 type %r 不支持（int/bool/float/string/list_int/list_float）"
                % (a["name"], t))
    return attr


def main():
    ap = argparse.ArgumentParser(description="板上执行单算子（ctypes + aclopExecuteV2）")
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--om", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)
    args = ap.parse_args()

    if args.repeat < 1:
        die("--repeat 至少 1")
    if args.warmup < 0:
        die("--warmup 不能为负")

    mpath = os.path.join(args.case_dir, "manifest.json")
    if not os.path.isfile(mpath):
        die("找不到 %s" % mpath)
    with open(mpath, "r", encoding="utf-8") as f:
        man = json.load(f)

    op_type = man["op_type"]
    print("算子 %s @ device %d  warmup=%d repeat=%d"
          % (op_type, args.device, args.warmup, args.repeat))

    acl = load_acl()
    check(acl.aclInit(None), "aclInit")
    keepalive = load_model(acl, args.om)          # 必须持有，见 load_model 注释
    check(acl.aclrtSetDevice(args.device), "aclrtSetDevice(%d)" % args.device)
    stream = c_vp()
    check(acl.aclrtCreateStream(ctypes.byref(stream)), "aclrtCreateStream")

    in_desc, in_buf, dev_ptrs, dim_arrays = [], [], [], []

    # ---- 输入：逐槽构造，缺席的占位（见文件头第 1 条）----
    for t in man["inputs"]:
        if t.get("absent"):
            desc = acl.aclCreateTensorDesc(ACL_DT_UNDEFINED, 0, None, ACL_FORMAT_UNDEFINED)
            if not desc:
                die("aclCreateTensorDesc(UNDEFINED, 缺席 optional %s) 返回 null" % t["name"])
            buf = acl.aclCreateDataBuffer(None, 0)
            if not buf:
                die("aclCreateDataBuffer(nullptr, 0) 返回 null")
            in_desc.append(desc)
            in_buf.append(buf)
            print("  in  %-18s （缺席 optional 占位）" % t["name"])
            continue

        p = os.path.join(args.case_dir, t["file"])
        if not os.path.isfile(p):
            die("缺输入文件 %s" % p)
        with open(p, "rb") as f:
            data = f.read()
        if len(data) != t["bytes"]:
            die("%s 大小不对：%d 字节，manifest 说 %d。\n"
                "  传输被截断，或 case 和 manifest 不是同一轮生成的。"
                % (t["file"], len(data), t["bytes"]))
        dev = c_vp()
        check(acl.aclrtMalloc(ctypes.byref(dev), len(data), ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc(%s, %d)" % (t["name"], len(data)))
        dev_ptrs.append(dev)
        buf_host = ctypes.create_string_buffer(data, len(data))
        check(acl.aclrtMemcpy(dev, len(data), ctypes.cast(buf_host, c_vp),
                              len(data), ACL_MEMCPY_HOST_TO_DEVICE),
              "aclrtMemcpy H2D(%s)" % t["name"])
        dims = (c_i64 * len(t["shape"]))(*[int(d) for d in t["shape"]])
        dim_arrays.append(dims)
        desc = acl.aclCreateTensorDesc(int(t["dtype_code"]), len(t["shape"]), dims, ACL_FORMAT_ND)
        if not desc:
            die("aclCreateTensorDesc(%s) 返回 null" % t["name"])
        buf = acl.aclCreateDataBuffer(dev, len(data))
        if not buf:
            die("aclCreateDataBuffer(%s) 返回 null" % t["name"])
        in_desc.append(desc)
        in_buf.append(buf)
        print("  in  %-18s %-9s %-22s %12d 字节"
              % (t["name"], t["dtype"], t["shape"], len(data)))

    # ---- 输出 ----
    out_desc, out_buf, out_dev, out_meta = [], [], [], []
    for t in man["outputs"]:
        dev = c_vp()
        check(acl.aclrtMalloc(ctypes.byref(dev), t["bytes"], ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc(%s, %d)" % (t["name"], t["bytes"]))
        out_dev.append(dev)
        dims = (c_i64 * len(t["shape"]))(*[int(d) for d in t["shape"]])
        dim_arrays.append(dims)
        desc = acl.aclCreateTensorDesc(int(t["dtype_code"]), len(t["shape"]), dims, ACL_FORMAT_ND)
        if not desc:
            die("aclCreateTensorDesc(%s) 返回 null" % t["name"])
        buf = acl.aclCreateDataBuffer(dev, t["bytes"])
        if not buf:
            die("aclCreateDataBuffer(%s) 返回 null" % t["name"])
        out_desc.append(desc)
        out_buf.append(buf)
        out_meta.append(t)
        print("  out %-18s %-9s %-22s %12d 字节"
              % (t["name"], t["dtype"], t["shape"], t["bytes"]))

    attr = build_attr(acl, man.get("attrs") or [])

    nin, nout = len(in_desc), len(out_desc)
    in_desc_a = (c_vp * nin)(*in_desc)
    in_buf_a = (c_vp * nin)(*in_buf)
    out_desc_a = (c_vp * nout)(*out_desc)
    out_buf_a = (c_vp * nout)(*out_buf)
    op_name = op_type.encode()

    def launch():
        return acl.aclopExecuteV2(op_name, nin, in_desc_a, in_buf_a,
                                  nout, out_desc_a, out_buf_a, attr, stream)

    # ---- warmup + repeat ----
    # 第一次下发会连带算子选择、模型匹配和首次 kernel 加载，比后面每次都慢得多。
    # 采 profiling 时 warmup 至少给 1，否则第一行数据明显偏大而看不出原因。
    for i in range(args.warmup):
        check(launch(), "aclopExecuteV2(warmup %d)" % i)
        check(acl.aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(warmup %d)" % i)
    if args.warmup:
        print("  warmup %d 次完成" % args.warmup)

    times = []
    for i in range(args.repeat):
        t0 = time.time()
        rc = launch()
        if rc != ACL_SUCCESS:
            hint = ""
            if rc == 100024:
                hint = ("\n  100024 = MatchOpModel 失败。模型是加载成功的，是**匹配**没过：\n"
                        "    - 算子名和装进 OPP 的那个不一致（差一个下划线也算）\n"
                        "    - 属性少设了一个，或值和 singleop.json 里的不同\n"
                        "    - optional 输入的槽被压缩了（必须原位占位）\n"
                        "  三者 ACL 都只报这一个码。")
            die("aclopExecuteV2 返回 %d%s" % (rc, hint))
        check(acl.aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(%d)" % i)
        times.append((time.time() - t0) * 1e6)

    # host 侧墙钟只能当量级参考 —— 真正的各 pipe 数据来自 msprof，由 host 侧解析。
    print("\n  host 墙钟（含下发与同步开销，仅供量级参考，单位 us）：")
    print("    min %.1f   mean %.1f   max %.1f   n=%d"
          % (min(times), sum(times) / len(times), max(times), len(times)))

    # ---- 取回输出 ----
    for dev, t in zip(out_dev, out_meta):
        host = ctypes.create_string_buffer(t["bytes"])
        check(acl.aclrtMemcpy(ctypes.cast(host, c_vp), t["bytes"], dev,
                              t["bytes"], ACL_MEMCPY_DEVICE_TO_HOST),
              "aclrtMemcpy D2H(%s)" % t["name"])
        p = os.path.join(args.case_dir, t["file"])
        with open(p, "wb") as f:
            f.write(host.raw)
        print("  输出 -> %s" % p)

    # ---- 收尾 ----
    for d in in_desc + out_desc:
        acl.aclDestroyTensorDesc(d)
    for b in in_buf + out_buf:
        acl.aclDestroyDataBuffer(b)
    for p in dev_ptrs + out_dev:
        acl.aclrtFree(p)
    acl.aclopDestroyAttr(attr)
    acl.aclrtDestroyStream(stream)
    acl.aclrtResetDevice(args.device)
    acl.aclFinalize()
    del keepalive
    print("\n[DEVICE] 完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
