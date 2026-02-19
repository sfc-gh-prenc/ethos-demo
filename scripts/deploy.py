"""Deploy a vLLM model via Ray Serve."""

import argparse
from pathlib import Path

import torch
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app


def main():
    parser = argparse.ArgumentParser(description="Deploy a vLLM model via Ray Serve")
    parser.add_argument(
        "--model-path", type=Path, required=True, help="Local path to model weights"
    )
    parser.add_argument("--model-id", type=str, default="ethos", help="Model ID for the API")
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs (default: all)")
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.95, help="GPU memory utilization"
    )
    args = parser.parse_args()

    num_gpus = args.num_gpus or torch.cuda.device_count()

    llm_config = LLMConfig(
        model_loading_config={"model_id": args.model_id, "model_source": str(args.model_path)},
        deployment_config={"autoscaling_config": {"min_replicas": 1, "max_replicas": 2}},
        engine_kwargs={
            "tensor_parallel_size": num_gpus,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
    )
    app = build_openai_app({"llm_configs": [llm_config]})
    serve.run(app, blocking=True)


if __name__ == "__main__":
    main()
