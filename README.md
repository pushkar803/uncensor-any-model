# Uncensor a chat model

Edit one behavior direction out of a causal language model and save the weights.

The notebook compares two prompt sets. At one depth it subtracts the average last-token state of ordinary prompts from the average last-token state of steered prompts. That unit vector is then removed from the token embedding, each attention output matrix, and each MLP down-projection.

## Files

| File | Purpose |
| --- | --- |
| `uncensor_model.ipynb` | The edit, from settings through save and optional Hub upload |
| `kernel-metadata.json` | Kaggle kernel settings: GPU, internet, private notebook |

## Run on Kaggle

The live notebook is [pushkarshetye/uncensor-qwen](https://www.kaggle.com/code/pushkarshetye/uncensor-qwen).

1. Open the notebook. In **Settings**, set **Accelerator** to **GPU T4 x2** and turn **Internet** on. Restart the session after changing either one.
2. To publish the result, create a Hugging Face access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). In the notebook, open **Add-ons → Secrets** and add it as `HF_TOKEN` or `HUB_TOKEN`. Attach the secret to the notebook.
3. Edit the settings cell if you want a different base model, sample size, depth, or spot-check prompt. Then run all cells.

`Qwen/Qwen2.5-3B-Instruct` in float16 fits one 16 GB T4. Edited weights are written to `/kaggle/working/<model-name>-edited`. A draft session drops that folder when it stops, so use **Save Version → Save & Run All** if you want the files kept as notebook output.

From this folder, after `kaggle kernels push` is configured for `pushkarshetye/uncensor-qwen`:

```bash
kaggle kernels push
```

## Settings

| Name | Default | What it does |
| --- | --- | --- |
| `base_model` | `Qwen/Qwen2.5-3B-Instruct` | Model to edit |
| `rows_per_side` | `128` | Prompts averaged on each side |
| `depth_fraction` | `0.65` | Where the direction is measured, as a fraction of depth |
| `edit_strength` | `1.0` | `1` removes the direction, `0` leaves the model unchanged |
| `spot_check` | a short prompt | Reply printed before and after the edit |
| `keep_private` | `True` | Hub repo is private |
| `upload_to_hub` | `True` | Upload after the local save when a token is attached |

Steered prompts come from the public AdvBench CSV. Ordinary prompts come from `tatsu-lab/alpaca`.

## Result

The Qwen 2.5 3B run is public:

https://huggingface.co/pushkar9867/Qwen2.5-3B-Instruct-edited

That repo contains `model.safetensors`, `config.json`, the tokenizer files, and `chat_template.jinja`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "pushkar9867/Qwen2.5-3B-Instruct-edited"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, device_map="auto")
```

New uploads from the notebook still follow `keep_private`. It defaults to `True`, so set it to `False` if the next run should be public too.

## Local secrets

`.env` is gitignored. Keep tokens there or in the Kaggle secret. Do not commit them.
