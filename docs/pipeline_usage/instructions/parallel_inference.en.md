# Pipeline Parallel Inference

## Specifying Multiple Inference Devices

For some pipelines in both the CLI and Python API, PaddleX supports specifying multiple inference devices simultaneously. If multiple devices are specified, at initialization each device will host its own instance of the underlying pipeline class, and incoming inputs will be inferred in parallel across them. For example, for the PP-StructureV3 pipeline:

```bash
paddlex --pipeline PP-StructureV3 \
        --input input_images/ \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --use_textline_orientation False \
        --save_path ./output \
        --device 'gpu:0,1,2,3'
```

```python
pipeline = create_pipeline(pipeline="PP-StructureV3", device="gpu:0,1,2,3")
output = pipeline.predict(
    input="input_images/",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
```

In both examples above, four GPUs (IDs 0, 1, 2, 3) are used to perform parallel inference on all files in the `input_images` directory.

When specifying multiple devices, the inference interface remains the same as when specifying a single device. Please refer to the pipeline usage guide to check whether a given pipeline supports multiple-device inference.

## Example of Multi-Process Parallel Inference

Beyond PaddleX’s built-in multi-GPU parallel inference, users can also implement parallelism by wrapping PaddleX pipeline API calls themselves according to their specific scenario, with a view to achieving a better speedup. Below is an example of using Python’s `multiprocessing` to run multiple cards and multiple pipeline instances in parallel over the files in an input directory:

```python
import argparse
import sys
from multiprocessing import Manager, Process
from pathlib import Path
from queue import Empty

from paddlex import create_pipeline
from paddlex.utils.device import constr_device, parse_device


def worker(pipeline_name_or_config_path, device, task_queue, batch_size, output_dir):
    pipeline = create_pipeline(pipeline_name_or_config_path, device=device)

    should_end = False
    batch = []

    while not should_end:
        try:
            input_path = task_queue.get_nowait()
        except Empty:
            should_end = True
        else:
            batch.append(input_path)

        if batch and (len(batch) == batch_size or should_end):
            try:
                for result in pipeline.predict(batch):
                    input_path = Path(result["input_path"])
                    if result.get("page_index") is not None:
                        output_path = f"{input_path.stem}_{result['page_index']}.json"
                    else:
                        output_path = f"{input_path.stem}.json"
                    output_path = str(Path(output_dir, output_path))
                    result.save_to_json(output_path)
                    print(f"Processed {repr(str(input_path))}")
            except Exception as e:
                print(
                    f"Error processing {batch} on {repr(device)}: {e}", file=sys.stderr
                )
            batch.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline", type=str, required=True, help="Pipeline name or config path."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory.")
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Specifies the devices for performing parallel inference.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory."
    )
    parser.add_argument(
        "--instances_per_device",
        type=int,
        default=1,
        help="Number of pipeline instances per device.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Inference batch size for each pipeline instance.",
    )
    parser.add_argument(
        "--input_glob_pattern",
        type=str,
        default="*",
        help="Pattern to find the input files.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"The input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if not input_dir.is_dir():
        print(f"{repr(str(input_dir))} is not a directory.", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        print(f"{repr(str(output_dir))} is not a directory.", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    device_type, device_ids = parse_device(args.device)
    if device_ids is None or len(device_ids) == 1:
        print(
            "Please specify at least two devices for performing parallel inference.",
            file=sys.stderr,
        )
        return 2

    if args.batch_size <= 0:
        print("Batch size must be greater than 0.", file=sys.stderr)
        return 2

    with Manager() as manager:
        task_queue = manager.Queue()
        for img_path in input_dir.glob(args.input_glob_pattern):
            task_queue.put(str(img_path))

        processes = []
        for device_id in device_ids:
            for _ in range(args.instances_per_device):
                device = constr_device(device_type, [device_id])
                p = Process(
                    target=worker,
                    args=(
                        args.pipeline,
                        device,
                        task_queue,
                        args.batch_size,
                        str(output_dir),
                    ),
                )
                p.start()
                processes.append(p)

        for p in processes:
            p.join()

    print("All done")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Assuming you save the script above as `mp_infer.py`, here are some example invocations:

```bash
# PP-StructureV3 pipeline
# Process all files in `input_images`
# Use GPUs 0,1,2,3 with 1 pipeline instance per GPU and batch size 1
python mp_infer.py \
    --pipeline PP-StructureV3 \
    --input_dir input_images/ \
    --device 'gpu:0,1,2,3' \
    --output_dir output1

# PP-StructureV3 pipeline
# Process all `.jpg` files in `input_images`
# Use GPUs 0 and 2 with 2 pipeline instances per GPU and batch size 4
python mp_infer.py \
    --pipeline PP-StructureV3 \
    --input_dir input_images/ \
    --device 'gpu:0,2' \
    --output_dir output2 \
    --instances_per_device 2 \
    --batch_size 4 \
    --input_glob_pattern '*.jpg'
```
