class Evaluter(object):
    def __init__(self, dataset, label_type):
        self.correct_preds, self.total_correct, self.total_preds = 0., 0., 0.

        self.right_tag, self.all_tag = 0

        self.label_type = label_type
        self.dataset = dataset

    def batch_update(self, batch, pred_labelid):
        (_, s_lengths), y = batch.word, batch.ner
        golden_lists = y.data.numpy()
        predict_lists = pred_labelid.data.numpy()
        for i, s_len in enumerate(s_lengths):
            golden_list = self.dataset.label_ids2labels(golden_lists[i], s_len)
            predict_list = self.dataset.label_ids2labels(predict_lists[i], s_len)

            for idy in range(s_len):
                if golden_list[idy] == predict_list[idy]:
                        self.right_tag += 1
            self.all_tag += len(golden_list)

            if self.label_type == "IOBES":
                gold_matrix = self.dataset.get_ner_IOBES(golden_list)
                pred_matrix = self.dataset.get_ner_IOBES(predict_list)
            else:
                gold_matrix = self.dataset.get_ner_BIO(golden_list)
                pred_matrix = self.dataset.get_ner_BIO(predict_list)

            right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
            self.correct_preds += len(right_ner)
            self.total_preds += len(pred_matrix)
            self.total_correct += len(gold_matrix)

    def get_metric(self):
        precision = self.correct_preds / self.total_preds if self.correct_preds > 0 else 0
        recall = self.correct_preds / self.total_correct if self.correct_preds > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if self.correct_preds > 0 else 0

        accuracy = self.right_tag/self.all_tag
        # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
        print("gold_num = ", self.total_correct, " pred_num = ", self.total_preds, " right_num = ", self.correct_preds)
        return accuracy, precision, recall, f1

def get_ner_IOBES(label_list):
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = None
    index_tag = None

    for i in range(0, list_len):
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
            index_tag = current_label.replace(begin_label,"",1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(single_label,"",1) +'[' +str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag +',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)


