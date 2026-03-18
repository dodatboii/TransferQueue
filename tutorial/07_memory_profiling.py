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

import os
import sys
import textwrap
import OmegaConf
import time
import warnings
from pathlib import Path

import ray  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

# Add the parent directory to the path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import transfer_queue as tq  # noqa: E402

# Configure Ray
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DEBUG"] = "1"

if not ray.is_initialized():
    ray.init(namespace="TransferQueueTutorial")


def present_data_workflow():
    """
    Demonstrate basic data workflow: put → get → clear.
    """
    print("=" * 80)
    print("Data Workflow: put → get → clear")
    print("=" * 80)

    # Step 1: Put data
    print("[Step 1] Putting data into TransferQueue...")
    tq_client = tq.get_client()

    input_ids = torch.randn(4096, 128000)

    data_batch = TensorDict(
        {
            "input_ids": input_ids
        },
        batch_size=input_ids.size(0),
    )

    print(f"  Created {data_batch.batch_size[0]} samples")
    partition_id = "partition_0"
    tq_client.put(data=data_batch, partition_id=partition_id)
    print(f"  ✓ Data written to partition: {partition_id}")

    # Step 2: Get metadata
    print("[Step 2] Requesting data metadata...")
    batch_meta = tq_client.get_meta(
        data_fields=["input_ids"],
        batch_size=4,  # 4是corner case，每个su拿一条；
        partition_id=partition_id,
        task_name="profiling_task",
    )
    print(f"  ✓ Got metadata: {len(batch_meta)} samples")
    print(f"    Global indexes: {batch_meta.global_indexes}")

    # Step 3: Get actual data
    print("[Step 3] Retrieving actual data...")
    retrieved_data = tq_client.get_data(batch_meta)
    print("  ✓ Data retrieved successfully")
    print(f"    Keys: {list(retrieved_data.keys())}")

    # Step 4: Clear
    print("===== before clear =====")
    time.sleep(10)
    print("[Step 4] Clearing partition... (you may also use clear_samples() to clear specific samples)")
    tq_client.clear_partition(partition_id=partition_id)
    print("  ✓ Partition cleared")
    print("=" * 80)
    print("Memory Profiling Complete! You can quit by ctrl+c now.")
    print("=" * 80)
    time.sleep(1000)


def main():
    try:
        print("Setting up TransferQueue...")
        config = OmegaConf.create(
            {
                "num_data_storage_units": 4,
            }
        )
        tq.init(config)

        print("Start workflow...")
        present_data_workflow()

        # Cleanup
        tq.close()
        ray.shutdown()
        print("\n✓ Cleanup complete")

    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
