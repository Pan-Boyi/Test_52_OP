#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""通用 1952（MC62 / 5102）单算子上板验证与性能剖析 —— host 侧驱动。

由两套内部单算子上板 harness 归纳而来：取其各自的长处，并把**算子相关的部分全部外移
到一个算子描述 JSON**，使这个脚本本身对算子无感知。

七个步骤，可单独跑也可连起来（--steps）：

    case    按算子描述生成输入 .bin 和 manifest.json
    om      写 singleop.json，调 atc 出 .om
    push    组 bundle（model/ + case/ + 设备侧脚本 + SHA256SUMS）传到板子
    run     在板上执行，可选套 msprof
    pull    把 PROF 目录和设备输出取回本地
    export  用 msprof.py export summary 把原始 PROF 解成 CSV
    parse   读 op_summary_*.csv，打各 pipe 的性能表

两个配置文件：
    opbench_config.json   机器与板子（soc、atc、remote、profile）—— 换算子不用改
    ops/<name>.json       算子描述（输入/输出/属性/填充规则）—— 换算子只改这个

================================================================================
从两个 harness 继承下来的、踩过的坑
================================================================================
下面每一条都不是风格偏好，是会让人白花一轮的东西。

1. **msprof 必须给绝对路径，不能搜 PATH。** 远端那个 ssh 是非交互 shell，读不到
   ~/.bashrc 里 source 的 set_env.sh，于是明明装了也报「PATH 里没有」；或者机器上有
   好几套 CANN，搜到的不是这次要用的那一套 —— 数据照出，来源不明。

2. **PROF 目录靠「跑之前/跑之后」的集合差认，不靠时间戳也不靠名字猜。** msprof 每次
   新建一个 PROF_* 目录，板上往往已经堆了很多。差集为空 = msprof 没产出（退出 2）,
   多于一个 = 并发或残留（退出 3），两种都必须当失败，不能随便挑一个。

3. **报最小值，不报平均。** host 侧抖动和别的进程抢核只会让某几次变慢，不会让它变快,
   所以最小值是对「这个 kernel 本身多快」掺杂噪声最少的估计。平均值和行数一并给出,
   用来判断抖动有多大。

4. **singleop.json 的属性必须一个不少地全写上。** ACL 是按属性的**值**匹配 .om 的,
   多一个少一个都匹配不上，而报出来是「算子没找到」(100024) —— 和真正原因毫无关系。

5. **缺席的 optional 输入不能省略**，要占位 {"format":"RESERVED","shape":[],
   "type":"UNDEFINED"}，否则后面的输入全部错位到前一个 IR 槽上。

6. **auth 写 "auto" 是个陷阱。** 有些板子禁掉了公钥登录只认密码，而本机没装 sshpass
   时 "auto" 会不声不响退回密钥 —— 表现是一句和认证毫无关系的
   「远端建目录失败 (rc=255)」。确定只能用密码时就写死 "password"。

7. **远端目录每次重建（rm -rf）。** 上一轮残留的 .om 会被 aclopSetModelDir 一起加载,
   然后按属性值匹配到旧的那个上去，静默跑出上一版的数。

8. **先探 ssh，再干本地重活。** 进不去的话没必要先花几十秒编 .om。
"""

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE_RUNNER = "opbench_device.py"

STEPS = ("case", "om", "push", "run", "pull", "export", "parse")

# op_summary 里的各 pipe 列。mac 就是 CUBE；fixpipe 在纯向量算子上恒为 0。
DEFAULT_COLUMNS = [
    "total_exe_time(us)",
    "mac_exe_time(us)",
    "scalar_exe_time(us)",
    "mte1_exe_time(us)",
    "mte2_exe_time(us)",
    "mte3_exe_time(us)",
    "fixpipe_time(us)",
]

REMOTE_DEFAULTS = {
    "host": "",
    "user": "",
    "port": 22,
    "dir": "",                 # 远端工作目录；脚本在它下面建 <op>/<case>/
    "password_env": "OPBENCH_REMOTE_PASSWORD",
    # "password" / "key" / "auto"。见文件头第 6 条。
    "auth": "auto",
    "identity": "",
    "cann_env": "",            # 板上要 source 的 set_env.sh；**空 = 保留板上现有环境**
    "device": 0,
    "repeat": 10,
    "warmup": 0,
    "ssh_opts": ["-o", "StrictHostKeyChecking=no",
                 "-o", "UserKnownHostsFile=/dev/null",
                 "-o", "LogLevel=ERROR"],
}

PROFILE_DEFAULTS = {
    "enabled": True,
    # 远端 msprof 的**绝对路径，写死**。见文件头第 1 条。
    "msprof": "/var/msprof",
    # 本地 TbeWorkTestSuit，用它的 analysis/msprof/msprof.py export summary。
    "tbe_work_test_suit": "",
    "columns": DEFAULT_COLUMNS,
}

# ACL 的 dtype 枚举和字节宽度。取自两个 harness 的设备侧（值已核对一致）。
DTYPE_CODE = {"float32": 0, "float16": 1, "int8": 2, "int32": 3,
              "uint8": 4, "int16": 6, "uint16": 7, "uint32": 8,
              "int64": 9, "uint64": 10, "double": 11, "bool": 12, "bfloat16": 27}
DTYPE_SIZE = {"float32": 4, "float16": 2, "int8": 1, "int32": 4,
              "uint8": 1, "int16": 2, "uint16": 2, "uint32": 4,
              "int64": 8, "uint64": 8, "double": 8, "bool": 1, "bfloat16": 2}
# atc 的 singleop.json 里 type 字段用的名字，和上面一致即可（atc 认这些小写名）。


def die(msg):
    print("\n[X] %s" % msg)
    sys.exit(1)


def step(msg):
    print("\n" + "=" * 72)
    print(msg)
    print("=" * 72)


def run(argv, cwd=None, env=None):
    print("  $ " + " ".join(shlex.quote(a) for a in argv))
    return subprocess.call(argv, cwd=cwd, env=env)


def run_out(argv, cwd=None):
    """跑并捕获 stdout；返回 (rc, stdout)。stderr 直通，便于看远端报错。"""
    print("  $ " + " ".join(shlex.quote(a) for a in argv))
    p = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE)
    out, _ = p.communicate()
    return p.returncode, out.decode("utf-8", "replace")


# ------------------------------------------------------------------ 配置

def load_json(path):
    if not os.path.isfile(path):
        die("找不到 %s" % path)
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except ValueError as e:
            die("%s 不是合法 JSON：%s" % (path, e))


ALLOWED_TOP = {"soc_version", "local_cann_env", "local_opp_env",
               "atc_bin", "atc_timeout", "remote", "profile"}


def check_unknown(doc, allowed, where):
    """未知键直接失败，不静默忽略。

    理由很实在：把 remote.dir 敲成 remote.directory
    之后，静默忽略的话脚本会拿着空 dir 继续走，最后报一句「远端建目录失败」，
    而真正的原因是键名拼错了。
    """
    unknown = sorted(set(doc) - allowed)
    if unknown:
        die("%s 有不认识的配置键：%s\n  可用的键：%s"
            % (where, ", ".join(unknown), ", ".join(sorted(allowed))))


def merge_defaults(doc, key, defaults):
    sub = doc.get(key) or {}
    check_unknown(sub, set(defaults), "%s.*" % key)
    out = dict(defaults)
    out.update(sub)
    return out


class Remote(object):
    """远端连接。**构造时不校验凭据** —— 只跑本地步骤（case/om/parse）时根本不需要
    它们，在这里就死掉会逼着人为了看一眼几何去配一遍密码。缺什么留到 preflight 说。
    """

    def __init__(self, cfg):
        self.host = cfg["host"]
        self.user = cfg["user"]
        self.port = int(cfg["port"])
        self.dir = cfg["dir"]
        self.auth = cfg["auth"]
        self.identity = cfg["identity"]
        self.cann_env = cfg["cann_env"]
        self.device = int(cfg["device"])
        self.repeat = int(cfg["repeat"])
        self.warmup = int(cfg["warmup"])
        self.ssh_opts = list(cfg["ssh_opts"])
        self.password = os.environ.get(cfg["password_env"], "")
        self.password_env = cfg["password_env"]

        self.missing = []
        if not self.host:
            self.missing.append("remote.host")
        if not self.user:
            self.missing.append("remote.user")
        if not self.dir:
            self.missing.append("remote.dir")

    @property
    def target(self):
        return "%s@%s" % (self.user, self.host)

    def _sshpass(self):
        """决定是否用 sshpass 包一层。见文件头第 6 条。"""
        if self.auth == "key":
            return []
        have = subprocess.call(["which", "sshpass"],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL) == 0
        if self.auth == "password":
            if not self.password:
                die("auth=password 但 $%s 是空的。\n"
                    "  设密码用 read，别让它进 shell 历史：\n"
                    "    read -rs %s && export %s"
                    % (self.password_env, self.password_env, self.password_env))
            if not have:
                die("auth=password 但本机没装 sshpass（apt install sshpass）")
            return ["sshpass", "-e", "-P", "assword"]
        # auto：有密码且有 sshpass 才用，否则走密钥。
        if self.password and have:
            return ["sshpass", "-e", "-P", "assword"]
        return []

    def _base_opts(self):
        opts = list(self.ssh_opts) + ["-p", str(self.port)]
        if self.identity:
            opts += ["-i", self.identity]
        return opts

    def ssh(self, script):
        env = dict(os.environ)
        if self.password:
            env["SSHPASS"] = self.password
        argv = self._sshpass() + ["ssh"] + self._base_opts() + [self.target, script]
        print("  $ ssh %s <<script>>" % self.target)
        return subprocess.call(argv, env=env)

    def ssh_out(self, script):
        env = dict(os.environ)
        if self.password:
            env["SSHPASS"] = self.password
        argv = self._sshpass() + ["ssh"] + self._base_opts() + [self.target, script]
        print("  $ ssh %s <<script>>" % self.target)
        p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE)
        out, _ = p.communicate()
        return p.returncode, out.decode("utf-8", "replace")

    def scp_to(self, local, remote):
        env = dict(os.environ)
        if self.password:
            env["SSHPASS"] = self.password
        opts = [o for o in self._base_opts()]
        # scp 用 -P 而不是 -p 指定端口；-p 在 scp 里是「保留时间戳」。
        opts = ["-P" if o == "-p" else o for o in opts]
        argv = self._sshpass() + ["scp", "-r"] + opts + [local, "%s:%s" % (self.target, remote)]
        print("  $ scp -r %s %s:%s" % (local, self.target, remote))
        return subprocess.call(argv, env=env)

    def scp_from(self, remote, local):
        env = dict(os.environ)
        if self.password:
            env["SSHPASS"] = self.password
        opts = ["-P" if o == "-p" else o for o in self._base_opts()]
        argv = self._sshpass() + ["scp", "-r"] + opts + ["%s:%s" % (self.target, remote), local]
        print("  $ scp -r %s:%s %s" % (self.target, remote, local))
        return subprocess.call(argv, env=env)


# 绝不允许作为远端工作目录的路径。push 这一步会对远端目录 rm -rf，所以这个清单是
# 防灾设施而不是洁癖：配错一个字（比如 dir 写成 "/"）就会在板子上删掉别人的东西。
# 这条护栏来自实际事故后补的校验。
UNSAFE_REMOTE_DIRS = {"/", "/root", "/home", "/tmp", "/usr", "/var", "/etc",
                      "/bin", "/sbin", "/lib", "/opt", "/boot", "/dev", "/proc", "/sys"}


def validate_remote(rt):
    """能连之前先把配置本身查一遍 —— 这些错都不该等到 ssh 之后才发现。"""
    if rt.auth not in ("auto", "password", "key"):
        die("remote.auth 只能是 auto / password / key，给的是 %r" % rt.auth)
    if rt.repeat < 1:
        die("remote.repeat 至少 1，给的是 %d" % rt.repeat)
    if rt.warmup < 0:
        die("remote.warmup 不能为负，给的是 %d" % rt.warmup)
    if not rt.dir.startswith("/"):
        die("remote.dir 必须是绝对路径，给的是 %r\n"
            "  相对路径会相对远端登录目录解析，而 push 要对它 rm -rf。" % rt.dir)
    norm = os.path.normpath(rt.dir).rstrip("/") or "/"
    if norm in UNSAFE_REMOTE_DIRS:
        die("remote.dir = %r 太危险，拒绝使用。\n"
            "  push 会对这个目录 rm -rf。用一个专属子目录，比如 /root/opbench。" % rt.dir)
    for cmd in ("ssh", "scp"):
        if subprocess.call(["which", cmd], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) != 0:
            die("本机没有 %s" % cmd)


def preflight(rt):
    """开工前先探一次远端。见文件头第 8 条。"""
    if rt.missing:
        die("要碰板子的步骤缺配置：%s" % ", ".join(rt.missing))
    validate_remote(rt)
    rc = rt.ssh("true")
    if rc != 0:
        hint = ""
        if rc == 255:
            hint = ("\n  rc=255 是 ssh 自己的失败（认证/网络/host key），不是远端命令失败。"
                    "\n  如果这块板禁了公钥登录，把 remote.auth 写死成 \"password\" 并导出 $%s。"
                    % rt.password_env)
        die("ssh 进不去 %s（rc=%d）%s" % (rt.target, rc, hint))
    print("  远端可达：%s" % rt.target)


# ------------------------------------------------------------------ 算子描述

def norm_tensor(t, idx, what):
    if t.get("absent"):
        return {"absent": True, "name": t.get("name", "%s%d" % (what, idx))}
    for k in ("name", "shape", "dtype"):
        if k not in t:
            die("%s[%d] 缺 %s" % (what, idx, k))
    dt = t["dtype"]
    if dt not in DTYPE_CODE:
        die("%s[%d] 的 dtype %r 不认识；支持：%s"
            % (what, idx, dt, ", ".join(sorted(DTYPE_CODE))))
    shape = [int(d) for d in t["shape"]]
    if any(d <= 0 for d in shape):
        die("%s[%d] (%s) 的 shape 有非正数：%s" % (what, idx, t["name"], shape))
    n = 1
    for d in shape:
        n *= d
    return {
        "absent": False,
        "name": t["name"],
        "shape": shape,
        "dtype": dt,
        "format": t.get("format", "ND"),
        "elems": n,
        "bytes": n * DTYPE_SIZE[dt],
        "fill": t.get("fill") or {"kind": "uniform", "lo": -1.0, "hi": 1.0},
    }


def load_op_spec(path):
    doc = load_json(path)
    if "op_type" not in doc:
        die("%s 缺 op_type" % path)
    ins = [norm_tensor(t, i, "inputs") for i, t in enumerate(doc.get("inputs") or [])]
    outs = [norm_tensor(t, i, "outputs") for i, t in enumerate(doc.get("outputs") or [])]
    if not ins:
        die("%s 的 inputs 是空的" % path)
    if not outs:
        die("%s 的 outputs 是空的" % path)
    if any(o["absent"] for o in outs):
        die("输出不能标 absent")
    return {
        "op_type": doc["op_type"],
        "inputs": ins,
        "outputs": outs,
        "attrs": doc.get("attrs") or [],
        "seed": int(doc.get("seed", 0)),
        "name": doc.get("name") or os.path.splitext(os.path.basename(path))[0],
    }


# ------------------------------------------------------------------ step: case

def gen_fill(np, spec, t):
    """按 fill 规则造一个 tensor。seed 由 (全局 seed, 名字) 决定 —— 换一个输入的
    规则不会把别的输入的数据也变掉，便于单独排查某一路。
    """
    f = t["fill"]
    kind = f.get("kind", "uniform")
    seed = (spec["seed"] * 1000003 + (hash(t["name"]) & 0xFFFF)) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)
    n = t["elems"]
    dt = t["dtype"]

    if kind == "zeros":
        a = np.zeros(n, dtype=np.float64)
    elif kind == "ones":
        a = np.ones(n, dtype=np.float64)
    elif kind == "const":
        a = np.full(n, float(f.get("value", 0.0)), dtype=np.float64)
    elif kind == "arange":
        a = float(f.get("start", 0.0)) + float(f.get("step", 1.0)) * np.arange(n, dtype=np.float64)
    elif kind == "normal":
        a = rng.normal(float(f.get("mean", 0.0)), float(f.get("std", 1.0)), n)
    elif kind == "randint":
        lo, hi = int(f.get("lo", 0)), int(f.get("hi", 2))
        a = rng.integers(lo, hi, n).astype(np.float64)
    elif kind == "uniform":
        lo, hi = float(f.get("lo", -1.0)), float(f.get("hi", 1.0))
        a = rng.uniform(lo, hi, n)
    else:
        die("输入 %s 的 fill.kind %r 不认识" % (t["name"], kind))

    np_dt = {"float32": np.float32, "float16": np.float16, "int8": np.int8,
             "int32": np.int32, "uint8": np.uint8, "int16": np.int16,
             "uint16": np.uint16, "uint32": np.uint32, "int64": np.int64,
             "uint64": np.uint64, "double": np.float64, "bool": np.uint8}
    if dt == "bfloat16":
        # numpy 没有 bf16：按 fp32 取高 16 位（截断，不做 round-to-nearest-even）。
        v = a.astype(np.float32).view(np.uint32)
        return (v >> 16).astype(np.uint16).tobytes()
    return a.astype(np_dt[dt]).tobytes()


def do_case(spec, paths):
    try:
        import numpy as np
    except ImportError:
        die("生成输入需要 numpy（host 侧；板子上不需要）")
    cd = paths["case"]
    os.makedirs(cd, exist_ok=True)
    manifest = {"op_type": spec["op_type"], "inputs": [], "outputs": [], "attrs": spec["attrs"]}
    total = 0
    for t in spec["inputs"]:
        if t["absent"]:
            manifest["inputs"].append({"name": t["name"], "absent": True})
            print("  %-18s （缺席 optional，占位）" % t["name"])
            continue
        fn = "in_%s.bin" % t["name"]
        blob = gen_fill(np, spec, t)
        if len(blob) != t["bytes"]:
            die("内部错：%s 生成了 %d 字节，应为 %d" % (t["name"], len(blob), t["bytes"]))
        with open(os.path.join(cd, fn), "wb") as f:
            f.write(blob)
        total += len(blob)
        manifest["inputs"].append({
            "name": t["name"], "absent": False, "file": fn, "shape": t["shape"],
            "dtype": t["dtype"], "dtype_code": DTYPE_CODE[t["dtype"]], "bytes": t["bytes"],
        })
        print("  %-18s %-9s %-22s %12d 字节  (%s)"
              % (t["name"], t["dtype"], t["shape"], t["bytes"], t["fill"].get("kind")))
    for t in spec["outputs"]:
        manifest["outputs"].append({
            "name": t["name"], "file": "out_%s.bin" % t["name"], "shape": t["shape"],
            "dtype": t["dtype"], "dtype_code": DTYPE_CODE[t["dtype"]], "bytes": t["bytes"],
        })
        total += t["bytes"]
        print("  %-18s %-9s %-22s %12d 字节  (输出)"
              % (t["name"], t["dtype"], t["shape"], t["bytes"]))
    with open(os.path.join(cd, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("\n  case 目录 %s，输入+输出合计 %.1f MB" % (cd, total / 1048576.0))


# ------------------------------------------------------------------ step: om

def build_singleop(spec):
    """atc --singleop 吃的描述。见文件头第 4、5 条。"""
    input_desc = []
    for t in spec["inputs"]:
        if t["absent"]:
            input_desc.append({"format": "RESERVED", "shape": [], "type": "UNDEFINED"})
        else:
            input_desc.append({"format": t["format"], "shape": t["shape"], "type": t["dtype"]})
    output_desc = [{"format": t["format"], "shape": t["shape"], "type": t["dtype"]}
                   for t in spec["outputs"]]
    node = {"op": spec["op_type"], "input_desc": input_desc, "output_desc": output_desc}
    if spec["attrs"]:
        node["attr"] = spec["attrs"]
    return [node]


def do_om(spec, paths, soc, atc_bin, atc_timeout, cann_env, opp_env):
    # 两个都 source：CANN runtime 和自定义算子包（OPP）。只 source 前者时 atc 跑得起来
    # 但找不到自定义算子，报的是「算子没注册」。
    if cann_env:
        source_env(cann_env)
    if opp_env:
        source_env(opp_env)
    os.makedirs(paths["omdir"], exist_ok=True)
    with open(paths["singleop"], "w", encoding="utf-8") as f:
        json.dump(build_singleop(spec), f, indent=2, ensure_ascii=False)
    print("  singleop.json -> %s" % paths["singleop"])

    # 先清掉旧 .om。atc 的输出名带形状签名，换形状会留下上一轮的文件，而
    # aclopSetModelDir 会把目录下所有 .om 一起加载（见文件头第 7 条）。
    for f in os.listdir(paths["omdir"]):
        if f.endswith(".om"):
            os.remove(os.path.join(paths["omdir"], f))

    atc = atc_bin or "atc"
    cmd = []
    if atc_timeout:
        cmd += ["timeout", "%ds" % int(atc_timeout)]
    cmd += [atc,
            "--singleop=" + paths["singleop"],
            "--soc_version=" + soc,
            "--output=" + paths["omdir"],
            "--log=error"]
    rc = run(cmd)
    if rc != 0:
        die("atc 返回 %d。失败原因 atc 只写进 ~/ascend/log/（OP_LOGE），stdout 看不到。" % rc)
    oms = [f for f in os.listdir(paths["omdir"]) if f.endswith(".om")]
    if not oms:
        die("atc 说成功了，但 %s 下没有 .om" % paths["omdir"])
    if len(oms) > 1:
        die("%s 下有 %d 个 .om：%s\n"
            "  设备侧只允许一个（aclopSetModelDir 会全部加载，然后按属性值匹配到旧的那个上去）"
            % (paths["omdir"], len(oms), ", ".join(sorted(oms))))
    print("  om: %s" % oms[0])
    return oms[0]


def source_env(path):
    """source 一个 set_env.sh 并把它改动的环境变量搬进当前进程。

    不能直接 subprocess 里 source —— 那只影响子进程。所以跑一次 bash 把 env 导出来再
    合并回来。
    """
    if not os.path.isfile(path):
        die("local_cann_env 不存在：%s" % path)
    script = "set -a; source %s >/dev/null 2>&1; env -0" % shlex.quote(path)
    p = subprocess.Popen(["bash", "-c", script], stdout=subprocess.PIPE)
    out, _ = p.communicate()
    if p.returncode != 0:
        die("source %s 失败" % path)
    for item in out.split(b"\0"):
        if not item:
            continue
        k, _, v = item.decode("utf-8", "replace").partition("=")
        os.environ[k] = v
    print("  已加载环境：%s" % path)


# ------------------------------------------------------------------ step: push

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def do_push(rt, paths, om_name):
    """组 bundle 并整体传过去。板上先校验 SHA256SUMS 再跑 —— 传大文件（这里 x 就有
    133 MB）偶发截断不是假想，而截断过的输入只会让结果不对，不会报错。
    """
    bundle = paths["bundle"]
    if os.path.isdir(bundle):
        import shutil
        shutil.rmtree(bundle)
    os.makedirs(os.path.join(bundle, "model"))
    os.makedirs(os.path.join(bundle, "case"))
    import shutil
    shutil.copy2(os.path.join(paths["omdir"], om_name), os.path.join(bundle, "model", om_name))
    for f in os.listdir(paths["case"]):
        shutil.copy2(os.path.join(paths["case"], f), os.path.join(bundle, "case", f))
    shutil.copy2(os.path.join(HERE, DEVICE_RUNNER), os.path.join(bundle, DEVICE_RUNNER))

    lines = []
    for root, _dirs, files in os.walk(bundle):
        for f in sorted(files):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, bundle)
            if rel == "SHA256SUMS":
                continue
            lines.append("%s  %s" % (sha256_file(p), rel))
    with open(os.path.join(bundle, "SHA256SUMS"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(lines)) + "\n")
    print("  bundle: %s（%d 个文件）" % (bundle, len(lines)))

    # 远端目录每次重建。见文件头第 7 条。
    rc = rt.ssh("set -e; rm -rf %s; mkdir -p %s" % (shlex.quote(rt.remote_dir),
                                                    shlex.quote(rt.remote_dir)))
    if rc != 0:
        die("远端建目录失败（rc=%d）" % rc)
    rc = rt.scp_to(bundle + "/.", rt.remote_dir)
    if rc != 0:
        die("scp 上传失败（rc=%d）" % rc)
    rc = rt.ssh("cd %s && sha256sum -c SHA256SUMS >/dev/null"
                % shlex.quote(rt.remote_dir))
    if rc != 0:
        die("远端 sha256sum 校验不过（rc=%d）—— 传输被截断了，重跑 push" % rc)
    print("  已上传并校验完整性")


# ------------------------------------------------------------------ step: run

def remote_script(rt, om_name, use_msprof, msprof, columns_hint):
    """板上执行脚本。msprof 模式下用集合差认出本次的 PROF 目录（见文件头第 2 条）。"""
    d = shlex.quote(rt.remote_dir)
    exec_cmd = ("python3 %s --case-dir case --om %s --device %d --repeat %d --warmup %d"
                % (DEVICE_RUNNER, shlex.quote("model/" + om_name),
                   rt.device, rt.repeat, rt.warmup))
    lines = ["set -u", "cd %s" % d]
    if rt.cann_env:
        lines += ["[ -f %s ] || { echo '[REMOTE ERROR] CANN_ENV 不存在: %s' >&2; exit 1; }"
                  % (shlex.quote(rt.cann_env), rt.cann_env),
                  "set +u", "source %s" % shlex.quote(rt.cann_env), "set -u"]
    else:
        lines += ["echo '[REMOTE] cann_env 为空 —— 保留板上现有环境' >&2"]

    if not use_msprof:
        lines += [exec_cmd, "RC=$?", "exit $RC"]
        return "\n".join(lines)

    q = shlex.quote(msprof)
    lines += [
        "MSPROF=%s" % q,
        "[ -x \"$MSPROF\" ] || { echo \"[REMOTE ERROR] $MSPROF 不存在或不可执行\" >&2; "
        "echo '  这个路径来自 opbench_config.json 的 profile.msprof，改那里' >&2; exit 127; }",
        "BEFORE=$(mktemp); AFTER=$(mktemp)",
        "trap 'rm -f \"$BEFORE\" \"$AFTER\"' EXIT",
        "find . -maxdepth 1 -mindepth 1 -type d -name 'PROF_*' -printf '%f\\n' | sort > \"$BEFORE\"",
        "\"$MSPROF\" " + exec_cmd + " >&2",
        "RC=$?",
        "find . -maxdepth 1 -mindepth 1 -type d -name 'PROF_*' -printf '%f\\n' | sort > \"$AFTER\"",
        "NEW=$(comm -13 \"$BEFORE\" \"$AFTER\")",
        "CNT=$(printf '%s' \"$NEW\" | grep -c . || true)",
        "if [ \"$CNT\" -eq 0 ]; then echo '[REMOTE ERROR] msprof 没产出 PROF_* 目录' >&2; exit 2; fi",
        "if [ \"$CNT\" -gt 1 ]; then echo \"[REMOTE ERROR] 本次多出 $CNT 个 PROF_* 目录，"
        "无法判断是哪一个（并发或残留）\" >&2; exit 3; fi",
        "if [ \"$RC\" -ne 0 ]; then echo \"[REMOTE 警告] 执行返回 $RC，但 PROF 已生成，继续\" >&2; fi",
        # stdout 只许有这一行：PROF 目录名。别的都走 stderr。
        "printf '%s\\n' \"$NEW\"",
    ]
    return "\n".join(lines)


def do_run(rt, om_name, use_msprof, msprof):
    script = remote_script(rt, om_name, use_msprof, msprof, None)
    if not use_msprof:
        rc = rt.ssh(script)
        if rc != 0:
            die("板上执行失败（rc=%d）" % rc)
        return None
    rc, out = rt.ssh_out(script)
    if rc == 2:
        die("msprof 没产出 PROF_* 目录。多半是 msprof 路径对但采集被拒"
            "（权限/设备被占），板上 stderr 在上面。")
    if rc == 3:
        die("本次多出不止一个 PROF_* 目录，没法确定是哪一个。"
            "清掉远端残留的 PROF_* 再跑，或确认没有并发任务。")
    if rc != 0:
        die("板上执行失败（rc=%d）" % rc)
    prof = out.strip().splitlines()
    prof = [l.strip() for l in prof if l.strip()]
    if len(prof) != 1:
        die("远端 stdout 应当只有一行 PROF 目录名，实际 %d 行：%r" % (len(prof), prof))
    print("  本次 PROF 目录：%s" % prof[0])
    return prof[0]


# ------------------------------------------------------------------ step: pull

def do_pull(rt, paths, prof_name):
    os.makedirs(paths["prof"], exist_ok=True)
    # 设备输出（给精度比对用）
    rc = rt.scp_from(rt.remote_dir + "/case/out_*.bin", paths["case"])
    if rc != 0:
        print("  [!] 没取到设备输出 out_*.bin（rc=%d）；只看性能的话可以忽略" % rc)
    if not prof_name:
        return None
    local = os.path.join(paths["prof"], prof_name)
    if os.path.isdir(local):
        import shutil
        shutil.rmtree(local)
    rc = rt.scp_from(rt.remote_dir + "/" + prof_name, paths["prof"])
    if rc != 0:
        die("取回 PROF 目录失败（rc=%d）" % rc)
    print("  PROF -> %s" % local)
    return local


# ------------------------------------------------------------------ step: export

def do_export(local_prof, suite):
    if not suite:
        die("没配 profile.tbe_work_test_suit，导不了 summary")
    if not os.path.isdir(suite):
        die("TbeWorkTestSuit 目录不存在：%s" % suite)
    msprof_py = os.path.join(suite, "analysis", "msprof", "msprof.py")
    if not os.path.isfile(msprof_py):
        die("找不到 %s" % msprof_py)
    # 用当前解释器，不写死路径 —— 装了多套 python 的机器上写死必错。
    rc = run([sys.executable, msprof_py, "export", "summary", "-dir", local_prof], cwd=suite)
    if rc != 0:
        die("msprof.py export summary 返回 %d" % rc)


# ------------------------------------------------------------------ step: parse

def find_op_summary(local_prof):
    d = os.path.join(local_prof, "mindstudio_profiler_output")
    if not os.path.isdir(d):
        die("没有 mindstudio_profiler_output：%s\n  先跑 export 这一步" % d)
    cands = [os.path.join(d, f) for f in os.listdir(d)
             if f.startswith("op_summary_") and f.endswith(".csv")]
    if not cands:
        die("%s 下没有 op_summary_*.csv" % d)
    cands.sort(key=os.path.getmtime, reverse=True)
    return cands[0]


def do_parse(csv_path, columns, op_type):
    # utf-8-sig：msprof 导出的 CSV 带 BOM，用 utf-8 读会让第一列名字多一个 ﻿,
    # 于是「缺列」而实际上列是在的。
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            die("CSV 没有表头：%s" % csv_path)
        missing = [c for c in columns if c not in reader.fieldnames]
        if missing:
            die("op_summary 缺列 %s\n  实际有的列：%s"
                % (missing, ", ".join(reader.fieldnames)))
        rows = list(reader)
    if not rows:
        die("op_summary 一行数据都没有：%s" % csv_path)

    # 只留目标算子的行。一次 launch 可能连带别的算子（比如框架插的 cast）。
    name_col = None
    for c in ("Op Name", "op_name", "Op Type", "op_type"):
        if c in (reader.fieldnames or []):
            name_col = c
            break
    if name_col:
        hit = [r for r in rows if op_type.lower() in str(r.get(name_col, "")).lower()]
        if hit:
            if len(hit) != len(rows):
                print("  （%d 行里有 %d 行属于 %s，其余已滤掉）" % (len(rows), len(hit), op_type))
            rows = hit

    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    rw = max(len("row"), len(str(len(rows))))
    header = ("%*s  " % (rw, "row")) + "  ".join("%*s" % (widths[c], c) for c in columns)
    print()
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows, 1):
        print(("%*d  " % (rw, i)) + "  ".join("%*s" % (widths[c], str(r.get(c, ""))) for c in columns))
    print("-" * len(header))

    stat = {"rows": len(rows)}
    for c in columns:
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(c, "")))
            except (TypeError, ValueError):
                pass
        if vals:
            stat[c] = {"min": min(vals), "mean": sum(vals) / len(vals), "max": max(vals)}

    # 见文件头第 3 条：报最小值，平均和最大一并给出以便看抖动。
    print("\n  各 pipe（%d 次下发，min / mean / max，单位 us）" % stat["rows"])
    w = max(len(c) for c in columns)
    for c in columns:
        s = stat.get(c)
        if not s:
            print("  %-*s   （没有可解析的数值）" % (w, c))
            continue
        jitter = (s["max"] - s["min"]) / s["min"] * 100.0 if s["min"] > 0 else 0.0
        print("  %-*s  %10.3f  %10.3f  %10.3f   抖动 %+.1f%%"
              % (w, c, s["min"], s["mean"], s["max"], jitter))
    return stat


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(
        description="通用 1952（MC62/5102）单算子上板验证与性能剖析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例：\n"
               "  %(prog)s ops/dec_enc_fused_kernel.json\n"
               "  %(prog)s ops/dec_enc_fused_kernel.json --steps case,om\n"
               "  %(prog)s ops/dec_enc_fused_kernel.json --steps parse\n"
               "  %(prog)s ops/dec_enc_fused_kernel.json --no-msprof --repeat 1\n")
    ap.add_argument("op_spec", help="算子描述 JSON（ops/*.json）")
    ap.add_argument("--config", default=os.path.join(HERE, "opbench_config.json"))
    ap.add_argument("--steps", default=",".join(STEPS),
                    help="要跑的步骤，逗号分隔。可选：" + ",".join(STEPS))
    ap.add_argument("--outdir", default="", help="产物根目录；默认 <配置目录>/out/<算子>")
    ap.add_argument("--soc", default="", help="覆盖 soc_version")
    ap.add_argument("--device", type=int, default=-1, help="覆盖 remote.device")
    ap.add_argument("--repeat", type=int, default=-1, help="覆盖 remote.repeat")
    ap.add_argument("--warmup", type=int, default=-1, help="覆盖 remote.warmup")
    ap.add_argument("--no-msprof", action="store_true", help="跑但不采集 profiling")
    ap.add_argument("--prof-dir", default="",
                    help="跳过 run/pull，直接解析已有的本地 PROF 目录名")
    args = ap.parse_args()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in STEPS]
    if bad:
        die("不认识的步骤 %s；可选：%s" % (bad, ", ".join(STEPS)))

    doc = load_json(args.config)
    check_unknown(doc, ALLOWED_TOP, "配置顶层")
    spec = load_op_spec(args.op_spec)
    soc = args.soc or doc.get("soc_version") or ""
    if "om" in steps and not soc:
        die("没给 soc_version。写进 opbench_config.json 或用 --soc（1952 是 MC62CM12AA）")

    rcfg = merge_defaults(doc, "remote", REMOTE_DEFAULTS)
    pcfg = merge_defaults(doc, "profile", PROFILE_DEFAULTS)
    if args.device >= 0:
        rcfg["device"] = args.device
    if args.repeat > 0:
        rcfg["repeat"] = args.repeat
    if args.warmup >= 0:
        rcfg["warmup"] = args.warmup
    rt = Remote(rcfg)

    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    out_root = args.outdir or os.path.join(cfg_dir, "out", spec["name"])
    paths = {
        "case": os.path.join(out_root, "case"),
        "omdir": os.path.join(out_root, "model"),
        "singleop": os.path.join(out_root, "singleop.json"),
        "bundle": os.path.join(out_root, "bundle"),
        "prof": os.path.join(out_root, "prof"),
    }
    rt.remote_dir = os.path.join(rt.dir, spec["name"]) if rt.dir else ""
    os.makedirs(out_root, exist_ok=True)

    use_msprof = pcfg.get("enabled", True) and not args.no_msprof

    print("算子 %s（%s）" % (spec["op_type"], spec["name"]))
    print("  soc      %s" % (soc or "(未配)"))
    print("  产物     %s" % out_root)
    print("  步骤     %s" % ",".join(steps))
    if {"push", "run", "pull"} & set(steps):
        print("  板子     %s:%s  device=%d repeat=%d warmup=%d"
              % (rt.target, rt.remote_dir, rt.device, rt.repeat, rt.warmup))
        print("  msprof   %s" % (pcfg["msprof"] if use_msprof else "关闭"))

    # 见文件头第 8 条。
    if {"push", "run", "pull"} & set(steps):
        step("远端预检")
        preflight(rt)

    om_name = None
    if "case" in steps:
        step("1 生成输入数据")
        do_case(spec, paths)
    if "om" in steps:
        step("2 atc 编译 .om")
        om_name = do_om(spec, paths, soc, doc.get("atc_bin", ""),
                        doc.get("atc_timeout", 1800), doc.get("local_cann_env", ""),
                        doc.get("local_opp_env", ""))
    if om_name is None and os.path.isdir(paths["omdir"]):
        oms = [f for f in os.listdir(paths["omdir"]) if f.endswith(".om")]
        if len(oms) == 1:
            om_name = oms[0]
        elif len(oms) > 1:
            die("%s 下有多个 .om，先跑 om 这一步重建" % paths["omdir"])

    if "push" in steps:
        step("3 组 bundle 并上传")
        if not om_name:
            die("没有 .om 可传；先跑 om 这一步")
        do_push(rt, paths, om_name)

    prof_name = args.prof_dir or None
    if "run" in steps:
        step("4 板上执行" + ("（msprof 采集）" if use_msprof else ""))
        if not om_name:
            die("不知道要跑哪个 .om；先跑 om 这一步")
        prof_name = do_run(rt, om_name, use_msprof, pcfg["msprof"])

    local_prof = None
    if "pull" in steps:
        step("5 取回结果")
        local_prof = do_pull(rt, paths, prof_name)
    elif prof_name:
        local_prof = os.path.join(paths["prof"], prof_name)

    if "export" in steps:
        if not local_prof:
            die("不知道解哪个 PROF 目录；用 --prof-dir <名字> 或先跑 run/pull")
        step("6 msprof export summary")
        do_export(local_prof, pcfg["tbe_work_test_suit"])

    if "parse" in steps:
        if not local_prof:
            # 只跑 parse 时自动挑最新的那个
            if os.path.isdir(paths["prof"]):
                ds = [os.path.join(paths["prof"], d) for d in os.listdir(paths["prof"])
                      if d.startswith("PROF_")]
                ds.sort(key=os.path.getmtime, reverse=True)
                if ds:
                    local_prof = ds[0]
                    print("  自动选用最新的 PROF 目录：%s" % os.path.basename(local_prof))
        if not local_prof:
            die("没有可解析的 PROF 目录")
        step("7 各 pipe 性能")
        do_parse(find_op_summary(local_prof), pcfg["columns"], spec["op_type"])

    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
