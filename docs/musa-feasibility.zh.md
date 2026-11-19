# DeepSelect MUSA 后端实现说明

## 许可证与第三方组件

DeepSelect 主项目沿用官方根目录 [`LICENSE`](../LICENSE) 中的 MIT License。
第三方库不并入主项目许可证文本，而是按官方仓库的方式通过
`.gitmodules` 固定引用，并保留各自子模块中的许可证文件：

- [MUTLASS](https://github.com/MooreThreads/mutlass)，路径为
  `csrc/3rdparty/mutlass`，固定提交由 gitlink 记录；其 BSD-3-Clause
  `LICENSE` 随子模块保留。
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)，路径为
  `csrc/3rdparty/cutlass`；其 `LICENSE.txt` 及适用的 EULA 随子模块保留。

初始化源码及第三方许可证的完整内容，请执行：

```bash
git submodule update --init --recursive
```

## 结论

公共 `deep_select.topk` 接口与 CUDA/Ascend 后端保持一致；除上游公共接口本身
不支持的 `begin` 和 `hint` 外，非空合法输入都由 MUSA C++/`.mu` kernel
处理，不回退到 `torch.topk`。

当前支持：

- BF16 和 FP32 输入；
- `topk=1..4096`；
- int32/int64 index；
- return value、用户提供的 index output buffer；
- 每行不同的 `end`、自定义越界填充值和 index offset；
- value 排序和 index 排序；
- NaN sentinel 与可选的 device assertion；
- MUSA graph capture/replay 和多 stream 并发；
- PEP 517 AOT wheel，以及源码检出的 JIT 诊断路径。

## 代码组织

- `deep_select/deep_select_musa.py`：公共参数检查和 MUSA dispatch；
- `deep_select/musa_cpp.py`：AOT 模块加载与源码检出的 JIT 诊断加载；
- `csrc/musa_kernels/`：BF16/FP32 原生 kernel；
- `tests/test_musa_backend.py`：公共 API 回归矩阵；
- `tests/test_musa_tme_survivor.py`：大 batch BF16 TME 专项回归；
- `benchmarks/`：Lightning Indexer 与 Sampling 场景 benchmark。

CUDA 和 Ascend 的 kernel、extension 与 dispatch 不因 MUSA 后端而改变。

## 选择算法

### BF16

BF16 映射为保持浮点数值顺序的 16-bit radix key。常规大 batch 路径使用
12-bit 首级 radix：第一次扫描确定 pivot，第二次扫描直接输出更大元素并
压缩边界桶，剩余低位只访问共享内存候选。边界候选溢出时执行精确 fallback，
因此低熵、全相等和重复值输入不会丢失元素。

低 batch 宽行使用多 CTA 分区和归约，以避免一个 CTA 无法充分利用设备。
scratch 按 MUSA stream 缓存，graph capture 前预热一次即可复用；不同 stream
不会共享可写 scratch。

对连续 BF16、`topk=512`、大 batch、超宽行，选择器使用 TME 将每个约 48 KB
tile 搬入共享内存，只保留当前精确 top-512 与高于 pivot 的新候选。候选重建
使用 8+8-bit 精确 radix，整行只从全局内存流式读取一次。该路径只在实测有收益
的 shape 范围启用；batch 不小于 2048 时使用 512 个 worker，其余使用 1024 个
worker。首 tile 的 NaN 检测与 high-byte histogram 融合为一次扫描；较窄输入继续
使用 compact12。

TME 头文件来自固定版本的公开
[`MooreThreads/mutlass`](https://github.com/MooreThreads/mutlass) submodule。
MUTLASS 的 BSD-3-Clause LICENSE 随 submodule 保留。

### FP32

FP32 常规大 batch 路径同样使用 12-bit 首级 radix，并通过对齐的 FP32x4
读取完成两次主扫描。剩余位在共享候选上解析；候选溢出时回到精确的多扫描
路径。低 batch 宽行使用多 CTA 的 4-byte radix 与分区归约。

Sampling 的 129280 宽度下，compact12 比基于 TME 的四级共享 radix 更快，
因此生产代码保留 compact12；TME 不会仅因为可用而被强制启用。

### Shortcut、排序和 NaN

当有效长度不大于 `topk` 时，kernel 直接复制有效前缀并写入用户指定的
index/value fill。一般变长行只读取 `end[row]` 之前的数据。

排序、offset 和 fill 均在原生扩展中完成。NaN 检测先写入与 CUDA 一致的
sentinel；启用 `abort_when_nan_found` 时由轻量 guard kernel 触发 device
assertion，避免影响无 NaN 的选择主路径。

## 正确性和安装验收

公共 API 回归覆盖：

- BF16/FP32 与多个 top-k 档位；
- int32/int64、return value、两种排序、`end`、offset 和 fill；
- 随机、低熵、全相等、重复值、正负零和 NaN 输入；
- 行长边界、非默认 output stride 和用户 output buffer；
- graph capture/replay 与两个 stream 并发；
- wheel 构建、隔离安装，并确认加载 wheel 内的 AOT `.so` 而非 JIT cache。

TME 专项回归使用能唯一识别期望 top-512 集合的输入，覆盖 int32/int64、
return value、sorted index、graph、多 stream 与 NaN sentinel。

大 shape 的精确性不以 `torch.topk` 作为唯一 oracle：回归会使用完整排序或
构造的已知 top-k 集合校验 selected values、index 合法性和 index 唯一性。

## 性能图

README 中两张 MUSA 图分别覆盖：

- Lightning Indexer：BF16、`topk=512`，batch 6/512/4096，vocab 16K–1M；
- Sampling：FP32、vocab 129280、`topk=512`、sorted values、return value、
  int64 index。

两张图都在同一次同设备对照中比较 DeepSelect 与 `torch.topk`，并在计时前
执行 cache 刷新。图的画布、颜色和坐标轴几何与 CUDA 参考对齐；MUSA y 轴固定
为 0–1 TB/s，以便读取实际差异。图和公开文档不记录内部主机、驱动、设备配置、
内部路径或 profiler 原始输出。

有效带宽按输入与输出的实际字节数除以 kernel 时间计算。不同平台的峰值带宽、
cache、atomic 和调度不同，因此图只用于同设备的实现对照，不将某个平台的绝对
TB/s 按峰值比例换算成另一平台结果。

## 安装

先初始化 submodule，再显式选择 MUSA 构建目标：

```bash
git submodule update --init --recursive
DEEP_SELECT_BUILD_TARGET_PLATFORM=MUSA python3 -m pip install .
```

源码检出下如果 AOT 模块不可用，`deep_select/musa_cpp.py` 可以 JIT 编译同一份
`.mu` 源用于诊断；发布 wheel 始终携带预编译的原生扩展。
