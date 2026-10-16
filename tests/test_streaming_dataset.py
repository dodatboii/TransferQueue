# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from transfer_queue.dataloader.streaming_dataloader import StreamingDataLoader
from transfer_queue.dataloader.streaming_dataset import StreamingDataset, chunk_batch_fn
from transfer_queue.metadata import BatchMeta


def test_chunk_batch_fn_dense():
    td = TensorDict(
        {"a": torch.arange(6).reshape(3, 2), "b": torch.arange(3).reshape(3, 1)},
        batch_size=(3,),
    )
    meta = BatchMeta(global_indexes=[0, 1, 2], partition_ids=["p"] * 3)
    chunks = chunk_batch_fn(td, meta, micro_batch_size=2)
    assert len(chunks) == 2
    assert chunks[0][0].batch_size == (2,)
    assert chunks[1][0].batch_size == (1,)
    assert torch.equal(chunks[0][0]["a"], torch.tensor([[0, 1], [2, 3]]))
    assert torch.equal(chunks[0][0]["b"], torch.tensor([[0], [1]]))
    assert torch.equal(chunks[1][0]["a"], torch.tensor([[4, 5]]))
    assert torch.equal(chunks[1][0]["b"], torch.tensor([[2]]))


def test_chunk_batch_fn_mixed_dense_and_jagged():
    values = [torch.tensor([1, 2]), torch.tensor([3, 4, 5]), torch.tensor([6])]
    nested = torch.nested.as_nested_tensor(values, layout=torch.jagged)
    td = TensorDict({"a": nested, "b": torch.arange(3)}, batch_size=(3,))
    meta = BatchMeta(global_indexes=[0, 1, 2], partition_ids=["p"] * 3)
    chunks = chunk_batch_fn(td, meta, micro_batch_size=2)
    assert len(chunks) == 2
    assert chunks[0][0].batch_size == (2,)
    assert chunks[1][0].batch_size == (1,)
    assert chunks[0][0]["a"].is_nested
    assert torch.equal(chunks[0][0]["a"][0], torch.tensor([1, 2]))
    assert torch.equal(chunks[0][0]["a"][1], torch.tensor([3, 4, 5]))
    assert torch.equal(chunks[0][0]["b"], torch.tensor([0, 1]))
    assert chunks[1][0]["a"].is_nested
    assert torch.equal(chunks[1][0]["a"][0], torch.tensor([6]))
    assert torch.equal(chunks[1][0]["b"], torch.tensor([2]))


def test_chunk_batch_fn_strided():
    values = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])]
    nested = torch.nested.as_nested_tensor(values, layout=torch.strided)
    td = TensorDict({"a": nested}, batch_size=(2,))
    meta = BatchMeta(global_indexes=[0, 1], partition_ids=["p"] * 2)
    chunks = chunk_batch_fn(td, meta, micro_batch_size=1)
    assert len(chunks) == 2
    assert chunks[0][0].batch_size == (1,)
    assert chunks[0][0]["a"].is_nested
    assert torch.equal(chunks[0][0]["a"][0], torch.tensor([[1, 2], [3, 4]]))
    assert chunks[1][0].batch_size == (1,)
    assert chunks[1][0]["a"].is_nested
    assert torch.equal(chunks[1][0]["a"][0], torch.tensor([[5, 6], [7, 8]]))


@pytest.fixture(params=[False, True], ids=["dataset", "dataloader"])
def replay_stream(request):
    dataset = StreamingDataset(
        config=OmegaConf.create({}),
        batch_size=2,
        micro_batch_size=1,
        data_fields=["value"],
        partition_id="round_1",
        task_name="train",
        dp_rank=0,
    )
    client = Mock()
    client.get_data.side_effect = lambda meta: TensorDict(
        {"value": torch.tensor(meta.global_indexes)}, batch_size=(meta.size,)
    )
    dataset._tq_client = client
    stream = StreamingDataLoader(dataset, num_workers=0) if request.param else dataset
    return dataset, stream, client


def test_reset_replays_cached_batches_without_fetching(replay_stream):
    dataset, stream, client = replay_stream
    client.get_meta.side_effect = [
        BatchMeta(global_indexes=[0, 1], partition_ids=["round_1"] * 2),
        BatchMeta(global_indexes=[2, 3], partition_ids=["round_1"] * 2),
    ]
    iterator = iter(stream)
    original = [next(iterator) for _ in range(4)]
    del iterator

    stream.reset()
    iterator = iter(stream)
    replayed = [next(iterator) for _ in range(4)]
    del iterator

    for (data, meta), (replay_data, replay_meta) in zip(original, replayed, strict=True):
        assert torch.equal(data["value"], replay_data["value"])
        assert meta.global_indexes == replay_meta.global_indexes
    assert len(dataset.buffer) == 2
    assert client.get_meta.call_count == 2
    assert client.get_data.call_count == 2
    client.reset_consumption.assert_not_called()


def test_step_fetches_from_new_partition_at_batch_zero(replay_stream):
    dataset, stream, client = replay_stream
    client.get_meta.side_effect = [
        BatchMeta(global_indexes=[0, 1], partition_ids=["round_1"] * 2),
        BatchMeta(global_indexes=[2, 3], partition_ids=["round_2"] * 2),
    ]
    iterator = iter(stream)
    next(iterator)
    next(iterator)
    del iterator

    stream.step("round_2")
    assert dataset.buffer == []
    assert dataset.batch_index == 0
    assert dataset.partition_id == "round_2"
    iterator = iter(stream)
    data, meta = next(iterator)
    del iterator

    assert data["value"].tolist() == [2]
    assert meta.partition_ids == ["round_2"]
    assert client.get_meta.call_args.kwargs["partition_id"] == "round_2"
    assert client.get_meta.call_args.kwargs["sampling_config"]["batch_index"] == 0
    assert [call[0] for call in client.mock_calls] == ["get_meta", "get_data", "get_meta", "get_data"]


def test_finite_stream_can_replay_after_consumption_completes(replay_stream):
    dataset, stream, client = replay_stream
    dataset.should_check_consumption_status = True
    client.check_consumption_status.return_value = False

    def fetch_meta(**kwargs):
        client.check_consumption_status.return_value = True
        return BatchMeta(global_indexes=[0, 1], partition_ids=["round_1"] * 2)

    client.get_meta.side_effect = fetch_meta
    original = list(stream)
    assert len(original) == 2
    assert len(dataset.buffer) == 1

    stream.reset()
    replayed = list(stream)
    assert [meta.global_indexes for _, meta in replayed] == [[0], [1]]
    assert client.get_meta.call_count == 1
    assert client.get_data.call_count == 1
    client.reset_consumption.assert_not_called()
