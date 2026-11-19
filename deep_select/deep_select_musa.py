"""MUSA backend for the public DeepSelect Python interface."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch_musa


_INPUT_ALIGNMENT_BYTES = 32
_OUTPUT_ALIGNMENT_BYTES = 32
_MAX_VOCAB_SIZE = 1 << 23
_NAN_SENTINEL = 0x3F3F3F3F
_EPILOGUE = os.environ.get("DEEP_SELECT_MUSA_EPILOGUE", "fused")
_EPILOGUE_MAX_ROWS = 1 << 30


def get_alignment_requirement() -> tuple[int, int]:
    return _INPUT_ALIGNMENT_BYTES, _OUTPUT_ALIGNMENT_BYTES


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _check_tensor(
    name: str,
    tensor: Optional[torch.Tensor],
    *,
    device: torch.device,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    contiguous: bool = False,
    last_dim_contiguous: bool = False,
) -> None:
    if tensor is None:
        return
    _check(tensor.device == device, f"{name} must be on {device}")
    _check(tuple(tensor.shape) == shape, f"{name} must have shape {shape}")
    _check(tensor.dtype == dtype, f"{name} must have dtype {dtype}")
    if contiguous:
        _check(tensor.is_contiguous(), f"{name} must be contiguous")
    if last_dim_contiguous:
        _check(tensor.stride(-1) == 1, f"{name}'s last dimension must be contiguous")


def _check_row_alignment(name: str, tensor: torch.Tensor, alignment: int) -> None:
    _check(
        tensor.stride(0) * tensor.element_size() % alignment == 0,
        f"{name}.stride(0) must be a multiple of {alignment} bytes",
    )


def _can_use_fast_path(
    input: torch.Tensor,
    topk: int,
    end: Optional[torch.Tensor],
    sorted_value: bool,
    sorted_index: bool,
    output_index: torch.Tensor,
    output_idx_offset: Optional[torch.Tensor],
    return_value: bool,
    abort_when_nan_found: bool,
) -> bool:
    if (
        input.dtype not in (torch.bfloat16, torch.float32)
        or input.shape[0] == 0
        or (sorted_value and input.dtype != torch.float32)
        or output_index.dtype not in (torch.int32, torch.int64)
        or input.stride(-1) != 1
        or output_index.stride(-1) != 1
    ):
        return False

    return True


def _launch_fast_path(
    input: torch.Tensor,
    output_index: torch.Tensor,
    output_value: Optional[torch.Tensor],
    sorted_value: bool,
    sorted_index: bool,
    output_idx_offset: Optional[torch.Tensor],
    abort_when_nan_found: bool,
    end: Optional[torch.Tensor],
    idx_oob_fill_value: int,
    value_oob_fill_value: float,
) -> None:
    from .musa_cpp import (
        launch_output_idx_offset,
        launch_sort_output,
        launch_sort_output_fused,
        launch_topk_bf16,
        launch_topk_fp32,
    )

    sorted_in_kernel = False
    if input.dtype == torch.bfloat16:
        launch_topk_bf16(
            input,
            output_index,
            output_value,
            end,
            idx_oob_fill_value,
            value_oob_fill_value,
            abort_when_nan_found,
        )
    else:
        sorted_in_kernel = launch_topk_fp32(
            input,
            output_index,
            output_value,
            end,
            idx_oob_fill_value,
            value_oob_fill_value,
            abort_when_nan_found,
            sort_by_value=sorted_value,
        )

    if (
        sorted_index
        and not sorted_value
        and _EPILOGUE == "fused"
        and output_index.shape[0] <= _EPILOGUE_MAX_ROWS
        and output_index.shape[1] <= 1024
        and output_idx_offset is not None
    ):
        launch_sort_output_fused(
            output_index, output_value, end, output_idx_offset, input.shape[1]
        )
    else:
        if (sorted_index or sorted_value) and not sorted_in_kernel:
            launch_sort_output(
                output_index,
                output_value,
                end,
                input.shape[1],
                sorted_value,
            )

        if output_idx_offset is not None:
            launch_output_idx_offset(
                output_index, end, output_idx_offset, input.shape[1]
            )


def _write_selected_rows(
    *,
    input: torch.Tensor,
    rows: torch.Tensor,
    valid_len: int,
    topk: int,
    sorted_value: bool,
    sorted_index: bool,
    output_value: Optional[torch.Tensor],
    output_index: torch.Tensor,
    output_idx_offset: Optional[torch.Tensor],
    abort_when_nan_found: bool,
) -> None:
    if rows.numel() == 0 or valid_len == 0:
        return

    source = input.index_select(0, rows)[:, :valid_len]
    if valid_len > topk:
        has_nan = torch.isnan(source).any(dim=1)
    else:
        has_nan = torch.zeros(rows.numel(), dtype=torch.bool, device=input.device)

    if abort_when_nan_found and bool(has_nan.any().item()):
        raise RuntimeError("DeepSelect MUSA: NaN found in the valid input range")

    nan_rows = rows[has_nan]
    if nan_rows.numel() > 0:
        output_index[nan_rows, 0] = _NAN_SENTINEL

    clean_rows = rows[~has_nan]
    if clean_rows.numel() == 0:
        return
    clean_source = source[~has_nan]
    selected_count = min(topk, valid_len)
    values, indices = torch.topk(
        clean_source,
        selected_count,
        dim=1,
        largest=True,
        sorted=sorted_value,
    )

    if sorted_index:
        indices, order = torch.sort(indices, dim=1)
        values = torch.gather(values, 1, order)

    if output_idx_offset is not None:
        indices = indices.to(torch.int64)
        indices = indices + output_idx_offset.index_select(0, clean_rows).to(
            torch.int64
        ).unsqueeze(1)

    output_index[clean_rows, :selected_count] = indices.to(output_index.dtype)
    if output_value is not None:
        output_value[clean_rows, :selected_count] = values


def _fallback_topk(
    *,
    input: torch.Tensor,
    topk: int,
    end: Optional[torch.Tensor],
    sorted_value: bool,
    sorted_index: bool,
    output_value: Optional[torch.Tensor],
    output_index: torch.Tensor,
    output_idx_offset: Optional[torch.Tensor],
    idx_oob_fill_value: int,
    value_oob_fill_value: float,
    abort_when_nan_found: bool,
) -> None:
    output_index.fill_(idx_oob_fill_value)
    if output_value is not None:
        output_value.fill_(value_oob_fill_value)

    batch_size, vocab_size = input.shape
    if end is None:
        has_nan = (
            torch.isnan(input).any(dim=1)
            if vocab_size > topk
            else torch.zeros(batch_size, dtype=torch.bool, device=input.device)
        )
        if abort_when_nan_found and bool(has_nan.any().item()):
            raise RuntimeError("DeepSelect MUSA: NaN found in the valid input range")

        selected_count = min(topk, vocab_size)
        values, indices = torch.topk(
            input,
            selected_count,
            dim=1,
            largest=True,
            sorted=sorted_value,
        )
        if sorted_index:
            indices, order = torch.sort(indices, dim=1)
            values = torch.gather(values, 1, order)
        if output_idx_offset is not None:
            indices = indices.to(torch.int64)
            indices = indices + output_idx_offset.to(torch.int64).unsqueeze(1)
        output_index[:, :selected_count] = indices.to(output_index.dtype)
        if output_value is not None:
            output_value[:, :selected_count] = values
        if not abort_when_nan_found and vocab_size > topk:
            output_index[has_nan, 0] = _NAN_SENTINEL
        return

    for valid_len in torch.unique(end).cpu().tolist():
        rows = torch.nonzero(end == valid_len, as_tuple=False).flatten().to(torch.int64)
        _write_selected_rows(
            input=input,
            rows=rows,
            valid_len=int(valid_len),
            topk=topk,
            sorted_value=sorted_value,
            sorted_index=sorted_index,
            output_value=output_value,
            output_index=output_index,
            output_idx_offset=output_idx_offset,
            abort_when_nan_found=abort_when_nan_found,
        )


def topk(
    input: torch.Tensor,
    topk: int,
    begin: Optional[torch.Tensor],
    end: Optional[torch.Tensor],
    sorted_value: bool,
    sorted_index: bool,
    output_value: Optional[torch.Tensor],
    output_index: torch.Tensor,
    output_idx_offset: Optional[torch.Tensor],
    idx_oob_fill_value: int,
    value_oob_fill_value: float,
    return_value: bool,
    abort_when_nan_found: bool,
) -> None:
    _check(input.device.type == "musa", "input must be a MUSA tensor")
    _check(input.dim() == 2, "input must be a 2-D tensor")
    batch_size, vocab_size = input.shape
    _check(isinstance(topk, int) and topk > 0, "topk must > 0")
    _check(topk <= 4096, "topk must be <= 4096")
    _check(vocab_size > 0, "vocab_size must be > 0")
    _check(vocab_size < _MAX_VOCAB_SIZE, "vocab_size must be < 2^23")
    _check(batch_size <= 0xFFFFFFFF, "batch size exceeds uint32 capacity")
    _check(begin is None, "`begin` is not supported currently")
    _check(
        not (sorted_value and not return_value),
        "`return_value` must be enabled when `sorted_value` is True",
    )
    _check(
        not (sorted_value and sorted_index),
        "`sorted_value` and `sorted_index` cannot be used at the same time",
    )
    _check(
        input.dtype in (torch.bfloat16, torch.float32),
        "input dtype must be bfloat16 or float32",
    )
    _check(
        not (sorted_value and input.dtype == torch.bfloat16),
        "`sorted_value` is only supported for float32 input",
    )

    _check_tensor(
        "input",
        input,
        device=input.device,
        shape=(batch_size, vocab_size),
        dtype=input.dtype,
        last_dim_contiguous=True,
    )
    _check_tensor(
        "end",
        end,
        device=input.device,
        shape=(batch_size,),
        dtype=torch.int32,
        contiguous=True,
    )
    _check_tensor(
        "output_value",
        output_value,
        device=input.device,
        shape=(batch_size, topk),
        dtype=input.dtype,
        last_dim_contiguous=True,
    )
    _check(
        output_index.dtype in (torch.int32, torch.int64),
        "output_idx dtype must be torch.int32 or torch.int64",
    )
    _check_tensor(
        "output_index",
        output_index,
        device=input.device,
        shape=(batch_size, topk),
        dtype=output_index.dtype,
        last_dim_contiguous=True,
    )
    _check_tensor(
        "output_idx_offset",
        output_idx_offset,
        device=input.device,
        shape=(batch_size,),
        dtype=torch.int32,
        contiguous=True,
    )
    _check(
        (return_value and output_value is not None)
        or (not return_value and output_value is None),
        "output_value presence must match return_value",
    )
    _check_row_alignment("input", input, _INPUT_ALIGNMENT_BYTES)
    _check_row_alignment("output_index", output_index, _OUTPUT_ALIGNMENT_BYTES)
    if output_value is not None:
        _check_row_alignment("output_value", output_value, _OUTPUT_ALIGNMENT_BYTES)

    if _can_use_fast_path(
        input,
        topk,
        end,
        sorted_value,
        sorted_index,
        output_index,
        output_idx_offset,
        return_value,
        abort_when_nan_found,
    ):
        _launch_fast_path(
            input,
            output_index,
            output_value,
            sorted_value,
            sorted_index,
            output_idx_offset,
            abort_when_nan_found,
            end,
            idx_oob_fill_value,
            value_oob_fill_value,
        )
        return

    _fallback_topk(
        input=input,
        topk=topk,
        end=end,
        sorted_value=sorted_value,
        sorted_index=sorted_index,
        output_value=output_value,
        output_index=output_index,
        output_idx_offset=output_idx_offset,
        idx_oob_fill_value=idx_oob_fill_value,
        value_oob_fill_value=value_oob_fill_value,
        abort_when_nan_found=abort_when_nan_found,
    )
