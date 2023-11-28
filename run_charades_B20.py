import os

# renew label -> train model -> test model -> ...
gpu_idx = 3
task, THRESOLD, B = "charades", 20, 20
for I in range(1, 10):
    SUFFIX = "B{}_THRESOLD{}_RE{}".format(B, THRESOLD, I)

    # ----------------- RENEW LABEL -----------------
    renew_cmd = "python update_label_charades.py {} {} {} {}".format(task, B, THRESOLD, I)
    print(renew_cmd)
    os.system(renew_cmd)
    print("----------------- RENEW LABEL -----------------\n\n")


    train_cmd = "python  main.py --task {} --max_pos_len 128 --char_dim 50 --suffix {} --gpu_idx {} --epochs 50".format(task, SUFFIX, gpu_idx)
    print("----------------- TRAIN MODEL -----------------")
    print(train_cmd)
    os.system("rm ./datapkl/{}_128_{}.pkl".format(task, SUFFIX))
    os.system(train_cmd)
    print("----------------- TRAIN MODEL ----------------- \n\n")

    test_cmd = train_cmd + " --mode test_save"
    print(test_cmd)
    os.system(test_cmd)
    print("----------------- TEST MODEL -----------------\n\n")
