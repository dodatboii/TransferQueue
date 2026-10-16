# StreamingDataLoader replay and partition lifecycle

`StreamingDataset` caches full fetched batches before splitting them into
micro-batches. Already yielded batches remain available for another training
pass. The cache grows until `step()` replaces it or the dataset is released;
finishing a finite iterator does not clear it.

| Operation | Local effect | Remote effect |
| --- | --- | --- |
| `reset()` | Rewind the cursor, preserving cached batches | None; consumption state is unchanged |
| `step(partition_id)` | Replace the buffer, reset the cursor, select a partition | None; old payloads and sampler state remain |
| `get_buffer()` | Return the parent's actual cache list, including yielded batches | None |

## Replaying a rollout

Use `num_workers=0` for lifecycle control through the loader. In this example,
`dataset` is configured for `rollout_001` in streaming mode. The caller knows
how many micro-batches make up a pass and finishes each fetched full batch
before moving to the next pass.

```python
loader = StreamingDataLoader(dataset, num_workers=0)

for epoch in range(num_training_passes):
    if epoch:
        loader.reset()
    iterator = iter(loader)
    for _ in range(micro_batches_per_pass):
        batch, metadata = next(iterator)
        train_micro_batch(batch, metadata)
    del iterator

loader.step("rollout_002")
# Create the next iterator to train on the new partition.
```

The first pass fetches and caches data. Later passes replay those local batches.
In streaming mode, replaying the last cached batch does not end iteration:
fetching resumes, so the caller must bound each pass. In finite mode
(`should_check_consumption_status=True`), iteration ends when the controller
reports consumption complete and the local buffer has been yielded. A later
`reset()` can still replay that buffer.

Finish or discard the current iterator before `reset()` or `step()`. Neither
method cancels an iterator that is partway through yielding micro-batches.
`step()` drops the dataset's reference to its old buffer; batches or buffer
lists retained by the caller can keep their data alive.

Switch to a new partition for the next rollout. Calling `step()` with the same
partition restarts batch numbering, while RankAwareSampler can still hold the
old results for those numbers. It is not a general way to release memory while
continuing the same stream. Remote partition cleanup must be coordinated
separately, after all consumers have finished.

## DataLoader workers

`reset()`, `step()`, and `get_buffer()` operate on the parent dataset. They do
not synchronize running worker copies, including persistent workers. A parent
buffer is not an aggregate of worker caches. The multi-worker tutorial consumes
a bounded number of steps and releases the loader; it does not demonstrate
cross-worker replay or live partition switching.

For an unbounded stream that keeps fetching without `step()`, cached payloads
continue to accumulate. This lifecycle documentation does not introduce cache
eviction or a no-cache mode.
