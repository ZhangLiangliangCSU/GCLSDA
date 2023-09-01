import numpy
import numpy as np
import pandas as pd
from base.recommender import Recommender
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from data.loader import FileIO
from os.path import abspath

from util.conf import ModelConf, OptionConf
from util.evaluation import ranking_evaluation
import sys

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set,i, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set, test_set,i, **kwargs)
        # 183!,data里面读出来id2item==183;id2user==409
        self.i = i
        self.data = Interaction(conf, training_set, test_set)
        self.bestPerformance = []
        top = self.ranking['-topN'].split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)
        # 保存获取映射
        np.save('user_id.npy',self.data.user)
        np.save('item_id.npy', self.data.item)

    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        # # print dataset statistics
        print('Training Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.training_size()))
        print('Test Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.test_size()))
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(self.data.test_set)


        # print(type(test_set))
        # path = 'dataset/mydata/neg_test_0.txt'
        # with open(path) as f:
        #     for line in f.readlines():
        #         line = line.strip().split()
        # 读取配置
        # conf = ModelConf('./conf/GCLSDA.conf')
        # test_set = FileIO.load_data_set(conf['test.set'], conf['model.type'])
        # # print(conf['test.set'])
        # test_name = conf['test.set'][-10:-4]
        # print(test_name)
        # path = f"./dataset/mydata/ran_test_{self.i}.txt"
        # with open(path) as f:
        #     for line in f.readlines():
        #         line = line.strip().split()
        #         snoRNA = line[0]
        #         disease = line[1]
        #         sno_id = self.data.get_user_id(snoRNA)
        #         dis_id = self.data.get_item_id(disease)
        #         if sno_id is not None and dis_id is not None:
        #             my_candidates = self.predict(snoRNA)
        #             # print(disease)
        #             # print(dis_id)
        #             my_score = my_candidates[int(dis_id)]
        #             act_list.append(int(line[2]))
        #             pre_list.append(my_score)
        # 获取预测
        act_list = []
        pre_list = []
        print(self.i)
        for line in self.data.test_data:
            snoRNA = line[0]
            disease = line[1]
            sno_id = self.data.get_user_id(snoRNA)
            dis_id = self.data.get_item_id(disease)
            if sno_id is not None and dis_id is not None:
                my_candidates = self.predict(snoRNA)
                # print(disease)
                # print(dis_id)
                my_score = my_candidates[int(dis_id)]
                act_list.append(int(line[2]))
                pre_list.append(my_score)
        act_list = np.array(act_list)
        pre_list = np.array(pre_list)
        fpr, tpr, thresholds = roc_curve(act_list, pre_list, pos_label=1)
        roc_auc = auc(fpr, tpr)
        # path = 're/{}e-{}l-test_{}.xlsx'.format(self.emb_size,self.n_layers,self.i)
        path0 = 're/auc/auc_{}.txt'.format(self.i)
        print('roc_auc: ', roc_auc)
        np.savetxt(path0,np.array([roc_auc]))
        # # 列表转为字典存入excel
        # out_dic = {'real':act_list,'pre':pre_list}
        # # print(out_dic)
        # pd.DataFrame(out_dic).to_excel(path, sheet_name='Sheet1', index=False)

        # plt.figure()
        # lw = 2
        # plt.plot(fpr, tpr, color='darkorange',
        #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.legend(loc="lower right")
        # plt.show()

        # print(self.data.test_set)
        # path2 = 're/matrix_{}'.format(self.i)

        for i, user in enumerate(self.data.test_set):
            # print('user:',user)
            candidates = self.predict(user)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            rated_list, li = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            # print(len(candidates))
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')

        # print(self.data.test_set)
        # numpy.save('origin_0.npy', self.data.test_set)
        # 对于验证集的预测结果，只有121个对应预测结果，test_0.txt有307个?集合去重
        # print(rec_list)

        # 获取结果预测矩阵
        # pd.DataFrame(rec_list).to_excel(path2, sheet_name='Sheet1', index=False)
        # numpy.save(path2,rec_list)
        return rec_list

    def evaluate(self, rec_list):
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
        for user in self.data.test_set:
            line = user + ':'
            for item in rec_list[user]:
                line += ' (' + item[0] + ',' + str(item[1]) + ')'
                if item[0] in self.data.test_set[user]:
                    line += '*'
            line += '\n'
            self.recOutput.append(line)
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        out_dir = self.output['-dir']
        file_name = self.config['model.name'] + '@' + current_time + '-top-' + str(self.max_N) + 'items' + '.txt'
        FileIO.write_file(out_dir, file_name, self.recOutput)
        print('The result has been output to ', abspath(out_dir), '.')
        file_name = self.config['model.name'] + '@' + current_time + '-performance' + '.txt'
        self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        self.model_log.add('###Evaluation Results###')
        self.model_log.add(self.result)
        FileIO.write_file(out_dir, file_name, self.result)
        print('The result of %s:\n%s' % (self.model_name, ''.join(self.result)))

    def fast_evaluation(self, epoch):
        print('Evaluating the model...')
        rec_list = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save()
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            self.bestPerformance.append(performance)
            self.save()
        print('-' * 120)
        print('Real-Time Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
        bp = ''
        # for k in self.bestPerformance[1]:
        #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + '  |  '
        bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + '  |  '
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + '  |  '
        # bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        print('-' * 120)
        return measure
