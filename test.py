import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
import glob
import gc

from utils_mod import tf_itr
from methods import *


def conv_pred(el, file):
    t = 10
    idx = np.argsort(el)[::-1]

    id_10 = file[idx[:t]]
    return ' '.join(['{}'.format(i) for i in id_10])


def predict(test_relative_path, submission_path, FOLDER, num=None):
    cate = pd.read_csv('Category.csv', sep=',')['Label_num'].values
    # LR
    # model = LR(3862)

    # shared-bottom/SB
    # model = SB(cate, [16], [2048, 4096])

    # SNR
    # model = SNR_Trans(cate, [16, 16] ,bottom_0_struct=[512,512,512,512,512,512,512,512],bottom_1_struct=[512,512,512,512,512,512,512,512],beta=0.9, zeta=1.1, gamma=-0.5)

    # CNN
    # model = CNN2(3862)

    # FCN
    model = FCN(3862)

    model.build((None, 1152))

    batch = 102400
    label_num = 3862
    wfn = sorted(glob.glob('weights/*.h5'))[-1]
    model.load_weights(wfn)
    print('loaded weight file: %s' % wfn)

    file = pd.read_csv('Mapping_out.csv', sep=',')['OldLabel'].values

    cnt = 0
    for d in tf_itr('', batch, label_num=label_num, FOLDER=FOLDER, num=num):
        cnt += 1
        idx, x1_val, x2_val, _ = d
        val_in = np.concatenate((x1_val, x2_val), axis=1)
        ypd = model.predict(val_in, verbose=1, batch_size=32)
        del x1_val, x2_val

        out = []
        for i in range(len(ypd)):
            out.append(conv_pred(ypd[i], file))

        df = pd.DataFrame.from_dict({'VideoId': idx, 'Label': out})

        if not os.path.exists(submission_path + '/tmp/' + test_relative_path):
            os.mkdir(submission_path + '/tmp/' + test_relative_path)

        df.to_csv(submission_path + '/tmp/' + test_relative_path + '/subm' + str(cnt) + '.csv', header=True,
                  index=False,
                  columns=['VideoId', 'Label'])
        gc.collect()

    f_subs = glob.glob(os.path.join(submission_path + '/tmp/' + test_relative_path + '/subm*.csv'))
    print(f_subs)
    df = pd.concat((pd.read_csv(f) for f in f_subs))
    df.to_csv(os.path.join(submission_path + '/11.csv'), index=None)


if __name__ == '__main__':
    submission_path = './output'

    #predict('', submission_path, 'train_validation')
    #predict('train200', submission_path, '../train_s')
    predict('valid100', submission_path, '../validation_s')
