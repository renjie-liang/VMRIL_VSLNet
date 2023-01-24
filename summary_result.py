import os
import re

from pprint import pprint


def get_iter_result(file_name):

    scores = {"train":[],
                "val":[],
                "test":[]}

    with open(file_name) as f:
        lines = f.readlines()


    # write scores
    for i in range(0, len(lines), 7):
        _, _, train, _, val, _, test = lines[i: i+7]
        scores["train"].append(train)
        scores["val"].append(val)
        scores["test"].append(test)

    iter_lines = []
    for tr, va, te in zip(scores["train"], scores["val"], scores["test"]):
        tr = tr.replace("\n", "\t")
        va = va.replace("\n", "\t")
        iter_lines.append(tr+va+te)

      
    # write "Best Epoch"
    for l in lines[::-1]:
        if l.startswith("Save Best Epoch"):
            a = re.findall("\d+", l)
            idx = int(a[0]) - 1
            iter_lines[idx] = iter_lines[idx].replace("\n", "\t") + "Best Epoch" + "\n"
            break
    return iter_lines


suffix = "./ckpt_t7/vslnet_tacos_new_128_rnn_weak30_6merge_iou10_best_re{}/model"
# suffix = "./ckpt_t7/vslnet_activitynet_new_128_rnn_weak30_6merge_iou10_fix_re{}/model"
# suffix = "./ckpt_t7/vslnet_charades_new_128_rnn_weak30_6merge_iou10_fix_re{}/model"

start, end = 1, 49

result_file = []
for i in range(start, end):
    ckdir = suffix.format(i)
    file_list = os.listdir(ckdir)
    file_list = [i for i in file_list if i[-3:]=="txt"]
    file_list = sorted(file_list)
    iter_result = os.path.join(ckdir, file_list[-1])
    result_file.append(iter_result)
pprint(result_file)



out_file = "./result/tmp.txt"
lines_out = []

for file_name in result_file:
    iter_lines = get_iter_result(file_name)
    lines_out.extend(iter_lines)
    lines_out.extend("\n")

with open(out_file, "w") as f:
    f.writelines(lines_out)

print("Done!")