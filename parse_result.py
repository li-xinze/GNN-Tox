### Parsing the result!
import os
import numpy as np
from tensorflow.compat.v1.train import summary_iterator

def get_test_acc(event_file):
    val_auc_list = np.zeros(100)
    test_auc_list = np.zeros(100)
    for e in list(summary_iterator(event_file)):
        if len(e.summary.value) == 0:
            continue
        if e.summary.value[0].tag == "data/val_auc":
            val_auc_list[e.step-1] = e.summary.value[0].simple_value
        if e.summary.value[0].tag == "data/test_auc":
            test_auc_list[e.step-1] = e.summary.value[0].simple_value

    best_epoch = np.argmax(val_auc_list)
    return test_auc_list[best_epoch]

if __name__ == "__main__":
    event_file_list = []
    seed_list = range(0, 10)
    all = []
    for i, seed in enumerate(seed_list):
        try:
            dir_name = "experiments/runs_tox21_f_dc_e1_15" + '/finetune_cls_runseed' + str(seed) + '/tox21/'
            file_in_dir = os.listdir(dir_name)
            for f in file_in_dir:
                if "events" in f:
                    event_file_list.append(dir_name + f)
        except:
            pass
    for i in event_file_list:
        print(get_test_acc(i))
        all.append(get_test_acc(i))
    print('-' * 20)
    all = np.array(all)
    all = all * 100
    print(np.around(np.mean(all), 1), np.around(np.std(all), 1))








