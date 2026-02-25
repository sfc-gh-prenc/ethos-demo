"""Deploy vLLM models via Ray Serve.

Deploys two models:
  - deepseek: deepseek-ai/DeepSeek-R1-Distill-Llama-70B on 2 GPUs
  - ethos: local model from --model-path on the remaining GPUs
"""

import argparse
from pathlib import Path

import torch
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

LLM_GPUS = 2


def main():
    parser = argparse.ArgumentParser(description="Deploy vLLM models via Ray Serve")
    parser.add_argument(
        "--model-path", type=Path, required=True, help="Local path to ethos model weights"
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.95, help="GPU memory utilization"
    )
    args = parser.parse_args()

    total_gpus = torch.cuda.device_count()
    llm_replicas = 3
    remaining_gpus = total_gpus - LLM_GPUS * llm_replicas

    if remaining_gpus < 1:
        raise RuntimeError(
            f"Need at least {LLM_GPUS + 1} GPUs (found {total_gpus}): "
            f"{LLM_GPUS} for LLMs + at least 1 for ETHOS"
        )

    # One ethos replica per remaining GPU
    ethos_replicas = remaining_gpus

    llm_config = LLMConfig(
        model_loading_config={
            "model_id": "llm/gpt-oss-120b",
            "model_source": "openai/gpt-oss-120b",
        },
        deployment_config={"num_replicas": llm_replicas},
        engine_kwargs={
            "tensor_parallel_size": LLM_GPUS,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
    )

    ethos_config = LLMConfig(
        model_loading_config={
            "model_id": f"ethos/{args.model_path.name.lower()}",
            "model_source": str(args.model_path),
        },
        deployment_config={"num_replicas": ethos_replicas},
        engine_kwargs={
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
    )

    app = build_openai_app({"llm_configs": [llm_config, ethos_config]})
    serve.run(app, blocking=True)


if __name__ == "__main__":
    main()
