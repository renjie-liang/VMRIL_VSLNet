# The code for "Partial Annotation-based Video Moment Retrieval via Iterative Learning" MM 2023



Note:
1. Here is a higer performance version based on SeqPAN  https://github.com/renjie-liang/VMRIL_SeqPAN.

## Download Feature
1. We use features on https://app.box.com/s/d7q5atlidb31cuj1u8znd7prgrck1r1s from https://github.com/26hzhang/SeqPAN.
2. You also can download the video features from https://huggingface.co/datasets/k-nick/NLVL. But the feature should be converted from h5 to .npy files. Or you can follow this repository to modify the load code. https://github.com/renjie-liang/TSGVZoo


## Quick Start
```
# train
# modify the feature_path and emb_path in main.py
python run_charades_P30.py

# summary the performance
python get_miou_P.py
```


### Citation
If you feel this project helpful to your research, please cite our work.
```
@inproceedings{ji2023partial,
  title={Partial annotation-based video moment retrieval via iterative learning},
  author={Ji, Wei and Liang, Renjie and Liao, Lizi and Fei, Hao and Feng, Fuli},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={4330--4339},
  year={2023}
}

```