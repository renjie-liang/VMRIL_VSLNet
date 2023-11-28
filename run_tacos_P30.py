import os

# renew label -> train model -> test model -> ...
gpu_idx = 3
task,  P,  T,  = "tacos", 30, 20
for I in range(1, 10):
    SUFFIX = "P{}_T{}_RE{}".format(P, T, I)

    # ----------------- RENEW LABEL -----------------
    renew_cmd = "python update_label_tacos_P.py {} {} {} {}".format(task, P, T, I)
    print("----------------- RENEW LABEL -----------------")
    print(renew_cmd)
    os.system(renew_cmd)

    train_cmd = "python  main.py --task {} --suffix {} --gpu_idx {}".format(task, SUFFIX, gpu_idx)
    print("----------------- TRAIN MODEL -----------------")
    print(train_cmd)
    os.system("rm ./datapkl/{}_128_{}.pkl".format(task, SUFFIX))
    os.system(train_cmd)
    
    test_cmd = train_cmd + " --mode test_save"
    print("----------------- TEST MODEL -----------------")
    print(test_cmd)
    os.system("rm ./results/{}/{}.pkl".format(task, SUFFIX))
    os.system(test_cmd)
