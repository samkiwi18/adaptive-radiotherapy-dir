# weights/

Weights are not stored in git (`*.pth` is ignored).

## Trained registration model

`best_reg_model.pth` is the checkpoint with the best validation TRE at the end of finetuning: 11.85 mm on the landmarks
of the three Learn2Reg validation patients, without test-time optimization. It is the starting point of every
test-time optimization (see *Quick start* and *Full workflow* in the [README](../README.md); training history in
`results/training/`).

| File | Size (bytes) | MD5 |
|---|---:|---|
| `best_reg_model.pth` | 256364750 | `79ef2c6f6664be5f5ec41b06e9df35c6` |

It is published as an asset of the GitHub Release `v1.0.0` of this repository. From the repository root:

```bash
curl -L -o weights/best_reg_model.pth \
  https://github.com/samcaggia/adaptive-radiotherapy-dir/releases/download/v1.0.0/best_reg_model.pth
md5sum weights/best_reg_model.pth   # 79ef2c6f6664be5f5ec41b06e9df35c6
```

`scripts/register_pair.py` (`--weights`) and the notebook (`WEIGHTS_PATH`) read it from `weights/best_reg_model.pth`.

## MONAI self-supervised weights

The SwinUNETR encoder starts from the self-supervised weights of Tang et al.
([tang2022selfsupervised](../REFERENCES.md#tang2022selfsupervised)), distributed by MONAI and not redistributed here.
`registration/model.py` downloads them at the first use from

<https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/ssl_pretrained_weights.pth>

and `torch.hub` keeps them in its cache. They are needed only to train from scratch (`build_components(...,
load_ssl=True)`); test-time optimization starts from `best_reg_model.pth`.
