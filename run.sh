# python main_t7_tacos.py  --suffix weak30_6merge_iou10_best_re2 --checkpoint ./ckpt_t7/vslnet_tacos_new_128_rnn_weak30_6merge_iou10_best_re0/model/best_ckpt.t7

# ************** charades **************
init :
python main.py  --task charades --suffix B30_RE0 --char_dim 50 --gpu_idx 3
python main.py  --task charades --suffix B30_RE0 --char_dim 50 --gpu_idx 3 --mode test_save

step 1:
python update_label.py charades 30 5 1
step 2:
python  main.py --task charades  --suffix B30_IOU5_RE1 --char_dim 50   --gpu_idx 1 --epochs 50
python  main.py --task charades  --suffix B30_IOU5_RE1 --char_dim 50   --gpu_idx 1 --epochs 50 --mode test_save


# ************** anet **************
init :
python main.py --task anet  --suffix P30_RE0 --char_dim 100 --gpu_idx 2

step 1:
python update_label.py anet 30 5 1





# ************** tacos **************
init :
python main.py --task tacos  --suffix P30_RE0 --gpu_idx 3
step 1:
python update_label.py anet 30 5 1