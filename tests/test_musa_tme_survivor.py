"""Focused public-API validation for the wide BF16 survivor selector."""

from __future__ import annotations

import torch
import torch_musa

import deep_select


BATCH = 512
WIDTH = 1048576
TOPK = 512
SENTINEL = 0x3F3F3F3F


def _assert_known_topk(input: torch.Tensor, indices: torch.Tensor) -> None:
    assert indices.shape == (BATCH, TOPK)
    assert torch.all(indices >= WIDTH - TOPK).item()
    assert torch.all(indices < WIDTH).item()
    ordered = torch.sort(indices, dim=1).values
    assert torch.all(ordered[:, 1:] != ordered[:, :-1]).item()
    selected = torch.gather(input, 1, indices.to(torch.int64))
    assert torch.all(torch.isposinf(selected)).item()


def main() -> None:
    generator = torch.Generator(device="musa")
    generator.manual_seed(20261102)
    input = torch.randn(
        BATCH,
        WIDTH,
        dtype=torch.bfloat16,
        device="musa",
        generator=generator,
    )
    input[:, -TOPK:] = float("inf")

    output_i32 = torch.empty(BATCH, TOPK, dtype=torch.int32, device="musa")
    deep_select.topk(
        input,
        TOPK,
        output_idx=output_i32,
        indices_type=torch.int32,
        return_value=False,
        abort_when_nan_found=False,
    )
    torch.musa.synchronize()
    _assert_known_topk(input, output_i32)

    output_i64 = torch.empty(BATCH, TOPK, dtype=torch.int64, device="musa")
    values, indices = deep_select.topk(
        input,
        TOPK,
        sorted_index=True,
        output_idx=output_i64,
        indices_type=torch.int64,
        return_value=True,
        abort_when_nan_found=False,
    )
    torch.musa.synchronize()
    assert indices.data_ptr() == output_i64.data_ptr()
    assert torch.all(torch.isposinf(values)).item()
    assert torch.all(indices[:, 1:] >= indices[:, :-1]).item()
    _assert_known_topk(input, indices)

    capture_stream = torch.musa.Stream()

    def graph_launch() -> None:
        deep_select.topk(
            input,
            TOPK,
            output_idx=output_i32,
            indices_type=torch.int32,
            return_value=False,
            abort_when_nan_found=False,
        )

    with torch.musa.stream(capture_stream):
        graph_launch()
    capture_stream.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=capture_stream):
        graph_launch()
    graph.replay()
    torch.musa.synchronize()
    _assert_known_topk(input, output_i32)

    streams = [torch.musa.Stream(), torch.musa.Stream()]
    stream_outputs = [
        torch.empty(BATCH, TOPK, dtype=torch.int32, device="musa")
        for _ in streams
    ]
    for _ in range(3):
        for stream, output in zip(streams, stream_outputs):
            with torch.musa.stream(stream):
                deep_select.topk(
                    input,
                    TOPK,
                    output_idx=output,
                    indices_type=torch.int32,
                    return_value=False,
                    abort_when_nan_found=False,
                )
    for stream in streams:
        stream.synchronize()
    for output in stream_outputs:
        _assert_known_topk(input, output)

    # Every tile overflows the bounded incoming buffer for this distribution.
    # The rebuild path must still return 512 distinct, valid indices.
    input.fill_(2.0)
    deep_select.topk(
        input,
        TOPK,
        output_idx=output_i32,
        indices_type=torch.int32,
        return_value=False,
        abort_when_nan_found=False,
    )
    torch.musa.synchronize()
    assert torch.all(output_i32 >= 0).item()
    assert torch.all(output_i32 < WIDTH).item()
    ordered = torch.sort(output_i32, dim=1).values
    assert torch.all(ordered[:, 1:] != ordered[:, :-1]).item()
    assert torch.all(torch.gather(input, 1, output_i32.to(torch.int64)) == 2.0).item()

    for nan_position in (0, WIDTH // 2):
        input.fill_(2.0)
        input[:, nan_position] = float("nan")
        deep_select.topk(
            input,
            TOPK,
            output_idx=output_i32,
            indices_type=torch.int32,
            return_value=False,
            abort_when_nan_found=False,
        )
        torch.musa.synchronize()
        assert torch.all(output_i32[:, 0] == SENTINEL).item()
    print("MUSA wide BF16 survivor validation passed")


if __name__ == "__main__":
    main()
