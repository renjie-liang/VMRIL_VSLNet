import os
import sys

def run(suffix, ckpt_in):
    out_train="./data/dataset/tacos_" + suffix + "/train.json"
    out_val="./data/dataset/tacos_" + suffix + "/val.json"
    out_test="./data/dataset/tacos_"+ suffix + "/test.json"
    os.system("rm {}".format(out_train))

    print("generate train label")
    cmd_generate_iter = "python iter_tacos_6merge_append.py --task tacos --fv new --in_file ./datasets_t7/tacos_new_128_weak30.pkl  --out_file {} --checkpoint {} --iou 10".format(out_train, ckpt_in)

    if not "merge" in suffix:
        cmd_generate_iter = cmd_generate_iter + " --no_bmn"
    print(cmd_generate_iter)
    os.system(cmd_generate_iter)
    os.system("cp ./data/dataset/tacos_weak30/val.json {}".format(out_val))
    os.system("cp ./data/dataset/tacos_weak30/test.json {}".format(out_test))
    print("train start")

    cmd_train = "python main_t7_tacos.py --task tacos --mode train --fv new  --suffix {} --checkpoint {} --epochs {}".format(suffix, ckpt_in, epoch)
    print(cmd_train)
    os.system(cmd_train)

def get_args(method, iter_numbers, epoch):
    epoch = str(epoch)
    suffix = method + "_re" + str(iter_numbers)

    ckpt_in="./ckpt_t7/vslnet_tacos_new_128_rnn_" + method + "_re{}".format(iter_numbers-1) + "/model/"  +"best_ckpt.t7"
    ckpt_out="./ckpt_t7/vslnet_tacos_new_128_rnn_" + method + "_re{}".format(iter_numbers) + "/model/"  +"best_ckpt.t7"
    return suffix, ckpt_in, ckpt_out

def init_rm(suffix, ckpt_out):
    os.system("rm {}".format(ckpt_out))
    pkl_file = os.path.join("./datasets_t7/", "tacos_new_128_{}.pkl".format(suffix))
    os.system("rm {}".format(pkl_file))



# python run_tacos.py weak30_iou10_fix
# python run_tacos.py weak30_6merge_iou10_best

method = sys.argv[1] 
epoch = 5
for i in range(1, 2):
    suffix, ckpt_in, ckpt_out = get_args(method, i, epoch)
    init_rm(suffix, ckpt_out)
    run(suffix, ckpt_in)