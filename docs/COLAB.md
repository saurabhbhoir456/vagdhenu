# Running Vāgdhenu on Google Colab

This guide runs Vāgdhenu in a Colab Python 3.14 virtual environment with PyTorch 2.14.1 and CUDA 12.6. Colab does not expose Python 3.14 as a notebook kernel by default, so the first setup cell uses `uv` to create a 3.14 environment and run the project inside it.

> The exact compatibility of every dependency with Python 3.14 should be checked on the first run. If the 3.14 environment cannot install an upstream package, use the project with Python 3.12 instead and pin the dependencies known to support that runtime.

## 1. Create a GPU runtime

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Choose **Runtime > Change runtime type**.
3. Set **Hardware accelerator** to **GPU**.
4. Run:

```python
import os
print(os.cpu_count())
```

Confirm that a GPU is available:

```python
!nvidia-smi
```

Colab GPU availability and quota vary by account. A T4, L4, or A100 is suitable; generation-specific VRAM and free-tier session limits determine whether the full demo completes.

## 2. Install `uv` and Python 3.14

Create a Python 3.14 virtual environment in `/content/vagdhenu-env`:

```bash
%%bash
set -e
python -m pip install --quiet --upgrade uv
uv python install 3.14
uv venv /content/vagdhenu-env --python 3.14
/content/vagdhenu-env/bin/python --version
```

Check that the virtual environment is Python 3.14 before continuing.

## 3. Clone the project

Colab starts in `/content`. Download the repository with Git:

```bash
%%bash
set -e
cd /content
if [ -d vagdhenu/.git ]; then
  cd vagdhenu
  git pull --ff-only
else
  git clone https://github.com/saurabhbhoir456/vagdhenu vagdhenu
fi
```

Replace the clone URL with the URL of your own fork if you are using modified source.

## 4. Install the runtime dependencies

Activate the environment and install the CUDA build of PyTorch, project dependencies, and BigVGAN:

```bash
%%bash
set -e
source /content/vagdhenu-env/bin/activate
cd /content/vagdhenu

python -m pip install --upgrade pip
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt

# BigVGAN is distributed as source, not as a Python package.
if [ ! -d BigVGAN/.git ]; then
  git clone --depth 1 https://github.com/NVIDIA/BigVGAN.git BigVGAN
fi
python scripts/download_weights.py

python -c 'import sys, torch; print(sys.version); print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
```

The final command prints the runtime and GPU status. Stop and verify CUDA is available before running inference. Do not combine `%pip` install commands with a shell-activated environment in the same Colab cell.

## 5. Render the bundled example

The bundled example uses a shard with a single Devanagari line:

```bash
%%bash
set -e
source /content/vagdhenu-env/bin/activate
cd /content/vagdhenu
export PYTHONPATH="/content/vagdhenu/BigVGAN:${PYTHONPATH:-}"
rm -rf /content/vagdhenu/out-colab

python src/render.py \
  --shard examples/sample_shard.json \
  --results /content/vagdhenu/out-colab/results.json \
  --outdir /content/vagdhenu/out-colab

ls -lh /content/vagdhenu/out-colab
```

The generated WAV is located at:

```text
/content/vagdhenu/out-colab/sample_anushtubh.wav
```

Play it in the notebook:

```python
from IPython.display import Audio

Audio('/content/vagdhenu/out-colab/sample_anushtubh.wav')
```

Download it if it is visible in the output of `ls`:

```python
from google.colab import files
files.download('/content/vagdhenu/out-colab/sample_anushtubh.wav')
```

## 6. Render custom text

Create a custom one-line JSON shard. The `meter` must match a meter entry available to the project, and `padas` should contain the Sanskrit text to render:

```python
import json
from pathlib import Path

shard = [{
    "id": "custom_verse",
    "meter": "anushtubh",
    "padas": ["यदा यदा हि धर्मस्य ग्लानिर्भवति भारत"],
    "seed": 60,
    "no_sandhi": True,
    "out": "out/custom_verse.wav",
}]

shard_path = Path('/content/vagdhenu/examples/colab_custom.json')
shard_path.write_text(json.dumps(shard, ensure_ascii=False), encoding='utf-8')
print(shard_path)
```

Render the shard in the virtual environment:

```bash
%%bash
set -e
source /content/vagdhenu-env/bin/activate
cd /content/vagdhenu
export PYTHONPATH="/content/vagdhenu/BigVGAN:${PYTHONPATH:-}"
rm -rf /content/vagdhenu/out-colab-custom

python src/render.py \
  --shard examples/colab_custom.json \
  --results /content/vagdhenu/out-colab-custom/results.json \
  --outdir /content/vagdhenu/out-colab-custom
```

Play or download the custom result:

```python
from IPython.display import Audio

output = '/content/vagdhenu/out-colab-custom/custom_verse.wav'
Audio(output)
```

```python
from google.colab import files
files.download(output)
```

## 7. Optional: persist downloads in Google Drive

The repository, environment, and model weights are deleted when a Colab runtime is recycled. To keep outputs and weights, mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then copy the completed output or point `CHAMP_ROOT` at a weights directory on Drive:

```bash
!mkdir -p /content/drive/MyDrive/vagdhenu-output
!cp /content/vagdhenu/out-colab-custom/custom_verse.wav /content/drive/MyDrive/vagdhenu-output/
```

To download weights into Drive instead of every runtime's local storage:

```bash
%%bash
source /content/vagdhenu-env/bin/activate
export VAGDHENU_HF=prathoshap/vagdhenu
cd /content/vagdhenu
mkdir -p /content/drive/MyDrive/vagdhenu-models
python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ['VAGDHENU_HF'],
    local_dir='/content/drive/MyDrive/vagdhenu-models',
)
PY
```

Set `CHAMP_ROOT` when rendering if the weights live on Drive:

```bash
CHAMP_ROOT=/content/drive/MyDrive/vagdhenu-models \
  python src/render.py --shard examples/colab_custom.json \
  --results /content/vagdhenu/out-colab-custom/results.json \
  --outdir /content/vagdhenu/out-colab-custom
```

A virtual environment can be copied to Drive, but it contains absolute runtime paths. It is generally safer to cache the weights and reinstall only dependencies after a runtime reset.

## 8. Run the demo application (optional)

The local Gradio UI can be launched from Colab. Install the ZeroGPU integration used by the demo, then start `app.py`:

```bash
%%bash
set -e
source /content/vagdhenu-env/bin/activate
cd /content/vagdhenu/demo
python -m pip install spaces gradio
export PYTHONPATH="/content/vagdhenu/BigVGAN:${PYTHONPATH:-}"
python app.py
```

Wait for Gradio to print a local URL. In Colab, use the **proxy URL** printed beside the local address rather than opening `localhost` directly. Keep this cell running while using the UI. Colab may terminate the process when the runtime disconnects.

## Troubleshooting

### `torch.cuda.is_available()` is false

- Choose **Runtime > Change runtime type > T4 GPU** and restart the runtime.
- Colab GPU limits are account-specific; select a GPU again after quota becomes available.
- Confirm that `nvidia-smi` shows a GPU.
- Do not mix CPU and CUDA wheels from different package indexes.

### `pip install` reports no matching package

Python 3.14 is new and upstream distributions may not publish a wheel for every dependency yet. Inspect the exact package and first verify whether source builds work. A Python 3.12 virtual environment is more likely to be supported by Colab and upstream ML packages:

```bash
%%bash
python -m pip install --quiet --upgrade uv
uv python install 3.12
uv venv /content/vagdhenu-py312 --python 3.12
source /content/vagdhenu-py312/bin/activate
python --version
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r /content/vagdhenu/requirements.txt
```

Do not run Python 3.12 and Python 3.14 setup cells together; create only the environment you plan to use.

### Dependency conflicts in IndicF5

Keep the pinned versions in `requirements.txt` and install the project as it is provided. Do not independently install a newer `transformers`, `accelerate`, or `x-transformers` release because IndicF5's import layout and inference API may change.

### `No module named f5_tts` or `bigvgan`

Activate `/content/vagdhenu-env` in the same shell cell that runs Python. Add the repository and BigVGAN clone to `PYTHONPATH`:

```bash
export PYTHONPATH="/content/vagdhenu:/content/vagdhenu/BigVGAN:${PYTHONPATH:-}"
```

### Hugging Face download fails

Public model weights normally do not require authentication. If Hugging Face rate-limits the session, retry from a new runtime or set a Hugging Face token through Colab's Secrets panel and authenticate with `huggingface_hub.login()`.

### Session becomes disconnected

Colab runtime recycling removes `/content` and all downloaded packages. Re-run the setup cells or copy model weights and outputs to Google Drive before disconnecting. Long jobs can reduce reliability in free tiers; split work into shorter stages and save outputs frequently.
