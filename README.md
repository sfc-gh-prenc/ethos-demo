# ethos-demo

Streamlit application for running text completion requests against a vLLM backend served via Ray Serve.

## Setup

```bash
make dev
```

## Deploy a model

Start the vLLM backend by pointing to a local model path:

```bash
python scripts/deploy.py --model-path /data/models/my-model
```

This uses all available GPUs by default. To override:

```bash
python scripts/deploy.py --model-path /data/models/my-model --num-gpus 4
```

All options:

| Flag                       | Default       | Description                           |
| -------------------------- | ------------- | ------------------------------------- |
| `--model-path`             | (required)    | Local path to model weights           |
| `--model-id`               | `ethos`       | Model ID exposed via the API          |
| `--num-gpus`               | all available | Number of GPUs for tensor parallelism |
| `--gpu-memory-utilization` | `0.95`        | Fraction of GPU memory to use         |

The deployment exposes an OpenAI-compatible API at `http://localhost:8000/v1`.

## Run the Streamlit app

Once the model is deployed:

```bash
ethos-app
```

Or equivalently:

```bash
streamlit run src/ethos_demo/app.py
```

The app lets you configure the endpoint URL, model ID, and number of requests from the sidebar, then visualizes per-request latency as a line chart.
