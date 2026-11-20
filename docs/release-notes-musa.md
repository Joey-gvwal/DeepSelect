# DeepSelect MUSA 0.0.1

首个 `deepselect-musa` 发布线（MUSA 后端打包 + 短序列 BF16 TopK 优化）。

## 安装

```bash
pip install --index-url https://moorethreads.github.io/deepselect-whl/musa52-torch211/ \
  "deepselect-musa==0.0.1+musa5.2.0torch2.11.0.post1"
python -m deep_select.compat
```

前置条件：MUSA SDK 5.2（`MUSA_HOME=/usr/local/musa`）、
`torch`/`torch_musa` 2.11.0.post1、Python 3.10、MTT S5000（`mp_31`）。

## 变更

- MUSA 后端：BF16/FP32 native TopK、AOT 预编译扩展、MUSA 平台探测。
- 短序列 BF16 TopK 专用路径（`rows<=8`、`width<=2048`、`topk<=1024`、
  `return_value=False`、连续行、无 `end`/NaN abort），其余情况回退通用路径。
- 统一 MUSA benchmark 入口：`benchmarks/musa_bench.py --suite short|wide|all`。
- BF16 NaN payload 检测修复（覆盖 signaling NaN）。
- 打包：确定性版本注入、local version wheel tag
  （`+musa5.2.0torch2.11.0.post1`）、构建 provenance、运行时兼容性检查。

## 兼容性

- 不支持 PyPI 直接安装二进制 wheel：PyPI 不接受 PEP 440 local version；
  预编译 wheel 通过上面的静态索引分发，PyPI 仅承载源码包（P2 开放）。
