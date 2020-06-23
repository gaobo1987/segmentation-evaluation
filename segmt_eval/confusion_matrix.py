
class ConfusionMatrix:
    def __init__(self):
        self.cm = {}

    def add_value_to_cell(self, row_label, col_label, val):
        key = (row_label, col_label)
        if key in self.cm:
            self.cm[key] += val
        else:
            self.cm[key] = val

    def get_value_from_cell(self, row_label, col_label):
        key = (row_label, col_label)
        return self.cm[key] if key in self.cm else None

    def row_labels(self):
        return list(set([k[0] for k in self.cm.keys()]))

    def col_labels(self):
        return list(set([k[1] for k in self.cm.keys()]))

    # total count per predicted label
    def row_label_counts(self):
        all_row_labels = self.row_labels()
        all_col_labels = self.col_labels()
        result = {}
        for rlbl in all_row_labels:
            count = 0
            for clbl in all_col_labels:
                val = self.get_value_from_cell(rlbl, clbl)
                if val is not None:
                    count += val
            result[rlbl] = count
        return result

    # total count per gold label
    def col_label_counts(self):
        all_row_labels = self.row_labels()
        all_col_labels = self.col_labels()
        result = {}
        for clbl in all_col_labels:
            count = 0
            for rlbl in all_row_labels:
                val = self.get_value_from_cell(rlbl, clbl)
                if val is not None:
                    count += val
            result[clbl] = count
        return result

    # true positive count per matched label
    def matched_label_counts(self):
        all_row_labels = self.row_labels()
        all_col_labels = self.col_labels()
        result = {}
        for lbl in all_row_labels:
            count = 0
            if lbl in all_col_labels:
                val = self.get_value_from_cell(lbl, lbl)
                if val is not None:
                    count += val

            result[lbl] = count

        for lbl in all_col_labels:
            if lbl not in result:
                result[lbl] = 0

        return result

    def precisions(self):
        matched_counts = self.matched_label_counts()
        row_counts = self.row_label_counts()
        result = {}
        for m in matched_counts.keys():
            TP = matched_counts[m]
            if m in row_counts:
                Total_Predicted = row_counts[m]
                result[m] = TP / Total_Predicted
            else:
                result[m] = 0
        return result

    def recalls(self):
        matched_counts = self.matched_label_counts()
        col_counts = self.col_label_counts()
        result = {}
        for m in matched_counts.keys():
            TP = matched_counts[m]
            if m in col_counts:
                Total_Gold = col_counts[m]
                result[m] = TP / Total_Gold
            else:
                result[m] = 0
        return result

    def f1s(self):
        precisions = self.precisions()
        recalls = self.recalls()
        result = {}
        for k in precisions.keys():
            p = precisions[k]
            r = recalls[k]
            if (p + r) > 0:
                f1 = 2 * p * r / (p + r)
            else:
                f1 = 0
            result[k] = f1
        return result

    def compute_sum(self):
        sum_ = 0
        for val in self.cm.values():
            sum_ += val
        return sum_

    def accuracy(self):
        trace = 0.
        for label in self.row_labels():
            if (label, label) in self.cm:
                trace += self.cm[(label, label)]
        return trace / self.compute_sum()

    def avg(self, _dict):
        count = len(_dict)
        if count > 0:
            _sum = 0
            for k in _dict.keys():
                _sum += _dict[k]
            return _sum / count
        else:
            return 0

    def wgt(self, _dict):
        count = len(_dict)
        if count > 0:
            gold_label_counts = self.col_label_counts()
            total = 0
            for k in gold_label_counts:
                total += gold_label_counts[k]

            _sum = 0
            for k in _dict.keys():
                gold_count = 0
                if k in gold_label_counts.keys():
                    gold_count = gold_label_counts[k]
                _sum += _dict[k] * gold_count
            return _sum / total
        else:
            return 0

    def avg_precision(self):
        return self.avg(self.precisions())

    def avg_recall(self):
        return self.avg(self.recalls())

    def avg_f1(self):
        return self.avg(self.f1s())

    def wgt_precision(self):
        return self.wgt(self.precisions())

    def wgt_recall(self):
        return self.wgt(self.recalls())

    def wgt_f1(self):
        return self.wgt(self.f1s())

    def show(self):
        all_row_labels = self.row_labels()
        all_col_labels = self.col_labels()
        delim = '\t'
        header = ''
        for c in all_col_labels:
            header += delim + c
        print(header)
        for r in all_row_labels:
            row = r
            for c in all_col_labels:
                val = self.get_value_from_cell(r, c)
                row += delim + str(val)
            print(row)
