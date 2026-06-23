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

from typing import Any

from tensordict import TensorDict

from transfer_queue.metadata import BatchMeta
from transfer_queue.storage.managers.base import KVStorageManager, StorageManagerFactory
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.perf_utils import IntervalPerfMonitor
from transfer_queue.utils.zmq_utils import ZMQServerInfo

logger = get_logger(__name__)


@StorageManagerFactory.register("Yuanrong")
class YuanrongStorageManager(KVStorageManager):
    """Storage manager for Yuanrong backend."""

    def __init__(self, controller_info: ZMQServerInfo, config: dict[str, Any]):
        worker_port = config.get("worker_port", None)
        client_name = config.get("client_name", None)

        if worker_port is None or not isinstance(worker_port, int):
            raise ValueError("Missing or invalid 'worker_port' in config")

        if client_name is None:
            logger.info("Missing 'client_name' in config, using default value('YuanrongStorageClient')")
            config["client_name"] = "YuanrongStorageClient"
        elif client_name != "YuanrongStorageClient":
            raise ValueError(f"Invalid 'client_name': {client_name} in config. Expecting 'YuanrongStorageClient'")
        super().__init__(controller_info, config)
        self._perf_monitor = IntervalPerfMonitor(caller_name=self.storage_manager_id)

    async def put_data(self, data: TensorDict, metadata: BatchMeta, data_parser=None) -> None:
        with self._perf_monitor.measure("PUT_DATA"):
            await super().put_data(data, metadata, data_parser)

    async def get_data(self, metadata: BatchMeta) -> TensorDict:
        with self._perf_monitor.measure("GET_DATA"):
            return await super().get_data(metadata)

    async def clear_data(self, metadata: BatchMeta) -> None:
        with self._perf_monitor.measure("CLEAR_DATA"):
            await super().clear_data(metadata)
