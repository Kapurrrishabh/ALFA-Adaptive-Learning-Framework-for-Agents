---
title: StockIntel Models
emoji: 📈
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
models:
  - amazon/chronos-bolt-small
  - NeoQuasar/Kronos-small
  - NeoQuasar/Kronos-Tokenizer-base
---

Model API for [StockIntel](https://github.com/Kapurrrishabh/financial-advisor). Choose **ZeroGPU** hardware in
Settings. The app calls it with `STOCKINTEL_MODEL_SPACE=<you>/stockintel-models`.
