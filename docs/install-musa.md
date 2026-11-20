# DeepSelect MUSA 安装指南（发布到 `docs/install-musa.md` 的草稿）

> 适用于 `deepselect-musa`（DeepSelect 的摩尔线程 MUSA 后端）。CUDA/Ascend 用户请走源码安装。

## 1. 前置条件

| 项 | 要求 |
| --- | --- |
| GPU | MTT S5000（offload arch `mp_31`） |
| 驱动 | 与 MUSA SDK 5.2 匹配的驱动（mccx-204 实测 `mthreads-gmi 2.4.2` / driver `5.2.0-server`） |
| MUSA SDK | 5.2，安装于 `/usr/local/musa`（`MUSA_HOME`），`mcc` 可用 |
| Python | 3.10 / 3.12（其它版本未验证） |
| torch / torch_musa | `2.11.0.post1+musa5.2.0`（必须成对；见 `docs/compatibility-musa52.json`） |

先自检环境：

```bash
mthreads-gmi | head
echo "$MUSA_HOME" ; which mcc ; mcc --version | head -1
python -c "import torch, torch_musa; print(torch.__version__, torch_musa.__version__)"
python -c "import torch; print(torch.musa.is_available())"   # 必须 True
```

## 2. 安装（三选一）

### 2.1 推荐：预编译 wheel（GitHub Pages 静态索引）

```bash
pip install --index-url https://moorethreads.github.io/deepselect-whl/musa52-torch211/ deepselect-musa
```

锁定到某个工具链组合：

```bash
pip install --index-url https://moorethreads.github.io/deepselect-whl/musa52-torch211/ \
  "deepselect-musa==0.0.1+musa5.2.0torch2.11.0.post1"
```

### 2.2 内网：共享 wheel 目录（与 torch_musa 现行方式一致）

```bash
# 目录由集群管理员维护，含 MANIFEST.txt（sha256）
ls /sglang-checkout/whl-52/deepselect/
pip install --no-index --no-deps /sglang-checkout/whl-52/deepselect/deepselect_musa-0.0.1+musa5.2.0torch2.11.0.post1-*.whl
```

### 2.3 PyPI：源码构建（**暂不可用，P2 修复 sdist 后开放**）

> 当前 sdist 不完整（缺 `csrc/musa_kernels`、mutlass 头文件与构建辅助模块），
> 从 sdist `pip install` 会失败。首发只提供 §2.1/§2.2 的预编译 wheel；
> sdist 修复后这里会改为：
>
> ```bash
> pip install deepselect-musa   # 需要 MUSA SDK + mcc，首次安装会编译
> ```

## 3. 安装后自检

```bash
python -m deep_select.compat        # 版本、设备、兼容性结论
python - <<'PY'
import torch, deep_select
x = torch.randn(4, 204800, dtype=torch.bfloat16, device="musa")
vals, idx = deep_select.topk(x, 1024)
print(vals.shape, idx.shape, vals.dtype, idx.dtype)
PY
```

## 4. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `ImportError: failed to load the backend ... deep_select_musa` | 装了 `deepselect-musa` 但环境不是 MUSA（`torch.musa.is_available()` False） | 确认驱动/GPU；`python -m deep_select.compat` |
| `RuntimeError: deepselect-musa was built for torch_musa 2.11..., but ... is installed` | torch/torch_musa 版本与 wheel 不匹配 | 按提示安装对应 local version 的 wheel，或升级/降级 torch_musa |
| `No matching distribution found for deepselect-musa` | 索引里没有当前 Python/组合的 wheel | 换 `--index-url` 的 variant 目录，或改用 sdist（PyPI） |
| 跑出 `device assertion` / NaN 相关 abort | 输入含 NaN 且 `abort_when_nan_found=True`（默认） | 语义如此；需要继续计算时传 `abort_when_nan_found=False` |
| `sorted_value` 与 `sorted_index` 同时为 True 报错 | 两者不能同时使用（MUSA 实现限制） | 只保留一个；参考 README |
| 首次 `pip install` 很慢 / 编译失败 | 走的是 PyPI sdist 源码构建 | 用 §2.1/§2.2 的预编译 wheel |

## 5. 卸载与版本查询

```bash
pip show deepselect-musa
pip uninstall deepselect-musa
```

兼容矩阵与发布历史见 `docs/compatibility-musa52.json` 与 GitHub Release（`MooreThreads/deepselect-whl`）。
