# VirPLM
## description
VirPLM is a two-stage framework that leverages the fine-tuned ESM-2 protein language model on H3N2 HA1 sequences for accurate antigenic prediction.
![VirPLM](figs/overview.jpg) 

## Getting Started
### dependencies
- Torch 1.11.0
- Transformers 4.47.1
- Tokenizers 0.21.0
- Safetensors 0.5.0
- Huggingface Hub
- Numpy 1.26.4
- Pandas 2.2.3
- Scipy 1.14.1
- Matplotlib 3.10.0


### How to run
If you want to manually setup VirPLM, we recommend you to use Anaconda to build the runtime environment. The ESM-2 setup is based on the [official ESM repository](https://github.com/facebookresearch/esm).

#### Step 1: Clone this repository from Github.
```bash
git clone https://github.com/xingyili/VirPLM.git
cd VirPLM
```

#### Step 2: Pretrain ESM-2 on the HA1 sequence set.
```bash
python main.py --mode pretrain --config configs/default.yaml \
  --fasta-paths data/Seq/H3N2_demo.fasta \
  --output-dir H3pre_model
```
#### Step 3: The downstream task supports two training strategies:

1. Cross-validation fine-tuning:

```bash
python main.py --mode cv --config configs/default.yaml \
  --pretrained-model-dir H3pre_model \
  --ha-path data/NAD/H3N2_demo_HA.csv \
  --hi-path data/NAD/H3N2_demo_HI.csv \
```

2. Retrospective time-split fine-tuning:

```bash
python main.py --mode backtest --config configs/default.yaml \
  --pretrained-model-dir H3pre_model \
  --ha-path data/NAD/H3N2_demo_HA.csv \
  --hi-path data/NAD/H3N2_demo_HI.csv \
  --train-year-num 7 \
  --test-year-num 1 \
  --min-test-start-year 2012
```

You can customize the execution by modifying `configs/default.yaml` or command-line arguments.
