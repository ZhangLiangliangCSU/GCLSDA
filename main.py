from SELFRec import SELFRec
from util.conf import ModelConf
import numpy as np
import torch


if __name__ == '__main__':
    # print(torch.__version__)
    # print(torch.version.cuda)
    # print(torch.backends.cudnn.version())
    # print(torch.cuda.is_available())

    model = 'GCLSDA'
    conf = ModelConf('./conf/' + model + '.conf')

    sum = 0
    k = 1
    for i in range(k):
        rec = SELFRec(conf,i)
        rec.execute()
        auc_path = 're/auc/auc_{}.txt'.format(i)
        auc = np.loadtxt(auc_path)
        sum += auc
    avg_auc = sum/k
    print('avg_auc: ',avg_auc)
    path = 're/auc/avg_auc.txt'
    np.savetxt(path, np.array([avg_auc]))
