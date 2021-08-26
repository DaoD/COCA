import numpy as np
import pytrec_eval

class Metrics(object):

    def __init__(self, score_file_path, segment=50):
        super(Metrics, self).__init__()
        self.score_file_path = score_file_path
        self.segment = segment

    def __read_socre_file(self, score_file_path):
        sessions = []
        one_sess = []
        with open(score_file_path, 'r') as infile:
            i = 0
            for line in infile.readlines():
                i += 1
                tokens = line.strip().split('\t')
                one_sess.append((float(tokens[0]), int(float(tokens[1]))))
                if i % self.segment == 0:
                    one_sess_tmp = np.array(one_sess)
                    if one_sess_tmp[:, 1].sum() > 0:
                        sessions.append(one_sess)
                    one_sess = []
        return sessions

    def evaluate_all_metrics(self):
        sessions = self.__read_socre_file(self.score_file_path)
        qrels = {}
        run = {}
        for idx, sess in enumerate(sessions):
            query_id = str(idx)
            if query_id not in qrels:
                qrels[query_id] = {}
            if query_id not in run:
                run[query_id] = {}
            for jdx, r in enumerate(sess):
                doc_id = str(jdx)
                qrels[query_id][doc_id] = int(r[1])
                run[query_id][doc_id] = float(r[0])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'recip_rank', 'ndcg_cut.1,3,5,10'})
        res = evaluator.evaluate(run)
        map_list = [v['map'] for v in res.values()]
        mrr_list = [v['recip_rank'] for v in res.values()]
        ndcg_1_list = [v['ndcg_cut_1'] for v in res.values()]
        ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]
        ndcg_5_list = [v['ndcg_cut_5'] for v in res.values()]
        ndcg_10_list = [v['ndcg_cut_10'] for v in res.values()]
        return (np.average(map_list), np.average(mrr_list), np.average(ndcg_1_list), np.average(ndcg_3_list), np.average(ndcg_5_list), np.average(ndcg_10_list))

if __name__ == '__main__':
    metric = Metrics('./output/aol/BertSessionSearch.aol.score_file.txt')
    result = metric.evaluate_all_metrics()
    for r in result:
        print(r)
