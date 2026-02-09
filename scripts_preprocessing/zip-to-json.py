import pyarrow.parquet as pq
import json

input_parquet = "train.parquet"
output_json = "train.json"

pf = pq.ParquetFile(input_parquet)

with open(output_json, "w", encoding="utf-8") as f:
    for batch in pf.iter_batches(batch_size=100_000):
        rows = batch.to_pylist()
        for row in rows:
            f.write(json.dumps(row) + "\n")

print("Chunked Parquet → JSON complete")
