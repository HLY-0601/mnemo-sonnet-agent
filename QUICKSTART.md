# Mnemo-Sonnet 快速上手

## 零依赖跑通
```bash
python scripts/run_demo.py
```

## 跑一轮进化
```bash
python scripts/evolve.py --tasks examples/tasks.json --golden examples/golden.json
```

## 真基座 LoRA
```bash
python scripts/train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --data data/evolution/sft.jsonl --out data/evolution/adapter_1
```

## 纯 numpy 模型
```bash
python scripts/train_numpy_model.py --steps 300 --out models/mnemo_sonnet_evo1.npz
python scripts/train_numpy_model.py --eval-only --model models/mnemo_sonnet_evo1.npz
```
