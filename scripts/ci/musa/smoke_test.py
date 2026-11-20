"""Minimal GPU smoke test for the prebuilt DeepSelect MUSA wheel.

Validated in the mccx-204 spike (2026-11-20): 6 representative shapes pass on
MTT S5000 with torch/torch_musa 2.11.0.post1+musa5.2.0.

Usage (inside the MUSA container, wheel already installed):
    python3 smoke_test.py \
        --expected-musa-suffix musa52 --expected-torch-musa 2.11.0.post1

Exit code 0 = pass. Prints a JSON report (CI archives it as evidence).
"""
import argparse
import json
import os
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expected-local-label", default="",
        help="e.g. musa5.2.0torch2.11.0.post1; checked against _build_info.LOCAL_LABEL",
    )
    parser.add_argument(
        "--expected-musa-suffix", default="",
        help="（兼容旧参数）e.g. musa5.2",
    )
    parser.add_argument(
        "--expected-torch-musa", default="",
        help="e.g. 2.11.0.post1; checked against deep_select._build_info.BUILD_TORCH_MUSA",
    )
    parser.add_argument("--output", default="", help="optional JSON output path")
    args = parser.parse_args()

    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "musa_available": bool(torch.musa.is_available()),
        "musa_device_count": int(torch.musa.device_count()),
    }
    try:
        import torch_musa

        report["torch_musa"] = torch_musa.__version__
    except Exception as exc:  # pragma: no cover
        report["torch_musa_error"] = repr(exc)
    try:
        import torchada

        report["torchada"] = getattr(torchada, "__version__", "?")
    except Exception as exc:  # pragma: no cover
        report["torchada_error"] = repr(exc)

    props = torch.musa.get_device_properties(0)
    report["device"] = props.name
    report["device_total_memory_gib"] = round(props.total_memory / 1024**3, 1)
    report["MTHREADS_VISIBLE_DEVICES"] = os.environ.get("MTHREADS_VISIBLE_DEVICES")

    import deep_select
    from deep_select.musa_cpp import _module

    report["deep_select_version"] = deep_select.__version__
    report["extension_file"] = getattr(_module(), "__file__", "?")
    report["deep_select_file"] = deep_select.__file__

    # 安装后必须加载 wheel 内的 AOT 扩展，而不是 JIT cache。
    report["loads_aot_extension"] = bool(
        report["extension_file"].endswith(".so")
        and "torch_extensions" not in report["extension_file"]
    )

    # P1 起 wheel 带 _build_info.py；旧 wheel 没有则标记 absent 但不判失败。
    ok = report["loads_aot_extension"]
    try:
        import deep_select._build_info as bi

        report["build_info"] = {
            k: getattr(bi, k)
            for k in (
                "PUBLIC_VERSION", "LOCAL_LABEL", "FULL_VERSION", "BUILD_TORCH", "BUILD_TORCH_PUBLIC", "BUILD_TORCH_MUSA",
                "BUILD_MUSA_SDK", "BUILD_GIT_HEAD", "BUILD_MUTLASS_COMMIT",
            )
        }
        if args.expected_local_label:
            match = bi.LOCAL_LABEL == args.expected_local_label
            report["expected_local_label_match"] = match
            ok = ok and match
        if args.expected_musa_suffix:
            in_local = args.expected_musa_suffix in bi.LOCAL_LABEL
            report["expected_musa_suffix_match"] = in_local
            ok = ok and in_local
        if args.expected_torch_musa:
            match = bi.BUILD_TORCH_MUSA.startswith(args.expected_torch_musa)
            report["expected_torch_musa_match"] = match
            ok = ok and match
    except ImportError:
        report["build_info"] = "absent (wheel built before _build_info.py was added)"

    cases = [
        (4, 204800, 1024, torch.bfloat16, {}),
        (1, 152064, 2048, torch.bfloat16, {}),
        (4, 129280, 1024, torch.bfloat16, {"sorted_index": True}),
        (2, 204800, 1024, torch.float32, {"sorted": True}),
        (1, 32768, 1024, torch.float32, {"sorted_index": True}),
        (8, 32768, 512, torch.bfloat16, {}),
    ]
    results = []
    for rows, vocab, k, dtype, kwargs in cases:
        name = f"{rows}x{vocab} topk={k} {str(dtype).replace('torch.', '')}{kwargs or ''}"
        entry = {"case": name}
        torch.musa.synchronize()
        x = torch.randn(rows, vocab, dtype=dtype, device="musa")
        values, indices = deep_select.topk(x, k, **kwargs)
        ref_values, _ref_idx = torch.topk(x, k, dim=-1)
        torch.musa.synchronize()

        # 1) values gathered by the returned indices must equal returned values
        gathered = x.gather(-1, indices.to(torch.int64))
        entry["gather_matches_values"] = bool(torch.equal(gathered, values))
        # 2) the selected value multiset must match torch.topk exactly
        ours_sorted = torch.sort(values.float(), dim=-1).values
        ref_sorted = torch.sort(ref_values.float(), dim=-1).values
        entry["values_match_torch_topk"] = bool(torch.equal(ours_sorted, ref_sorted))
        entry["value_max_abs_diff"] = float((ours_sorted - ref_sorted).abs().max().item())
        # 3) indices must be valid and unique per row
        entry["indices_in_range"] = bool(
            (indices.min().item() >= 0) and (indices.max().item() < vocab)
        )
        entry["indices_unique_per_row"] = bool(
            (indices.to(torch.int64).sort(dim=-1).values.diff(dim=-1) > 0).all().item()
        )
        entry["values_shape"] = list(values.shape)
        entry["indices_dtype"] = str(indices.dtype)
        if kwargs.get("sorted"):
            entry["sorted_values"] = bool(
                (values.float().diff(dim=-1) <= 0).all().item()
            )
        if kwargs.get("sorted_index"):
            # sorted_index == returned indices are ascending along the last dim
            entry["sorted_index_ascending"] = bool(
                (indices[:, 1:] >= indices[:, :-1]).all().item()
            )
        entry_ok = all(bool(v) for key, v in entry.items() if isinstance(v, bool))
        entry["ok"] = entry_ok
        ok = ok and entry_ok
        results.append(entry)
        del x, values, indices, ref_values
        torch.musa.empty_cache()

    report["cases"] = results
    report["all_passed"] = ok
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        with open(args.output, "w") as f:
            f.write(payload + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
