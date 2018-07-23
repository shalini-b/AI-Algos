import sys
# import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, f1_score, recall_score


class ClassifyText(object):
    def __init__(self, training_data, test_data):
        self.training_data = training_data
        self.test_data = test_data

    def vectorize_by_cnt(
            self,
            training_data=None,
            test_data=None,
            lowercase=False,
            stop_words=None,
            unigram=True):

        if unigram:
            # Unigram baseline
            _min, _max = 1, 1
        else:
            # Bigram baseline
            _min, _max = 1, 2

        if not training_data:
            training_data = self.training_data.data
        if not test_data:
            test_data = self.test_data.data

        cnt_vectorizer = CountVectorizer(
            lowercase=lowercase,
            stop_words=stop_words,
            ngram_range=(_min, _max))
        cnt_vectorizer_training = cnt_vectorizer.fit_transform(training_data)
        cnt_vectorizer_test = cnt_vectorizer.transform(test_data)
        return cnt_vectorizer_training, cnt_vectorizer_test

    def calculate_scores(self, y_predict):
        precision = precision_score(self.test_data.target, y_predict, average="macro")
        recall = recall_score(self.test_data.target, y_predict, average="macro")
        f1 = f1_score(self.test_data.target, y_predict, average="macro")
        return round(precision, 3), round(recall, 3), round(f1, 3)

    def explore_classifiers(self):
        result_data = []
        unigram_cnt_vectorizer_training, unigram_cnt_vectorizer_test = self.vectorize_by_cnt()
        bigram_cnt_vectorizer_training, bigram_cnt_vectorizer_test = self.vectorize_by_cnt(unigram=False)

        # Multinomial NB for unigram
        clf = MultinomialNB()
        clf.fit(unigram_cnt_vectorizer_training, self.training_data.target)
        y_predict = clf.predict(unigram_cnt_vectorizer_test)
        precision, recall, f1 = self.calculate_scores(y_predict)
        result_data.append(['NB', 'UB', precision, recall, f1])

        # Multinomial NB for bigram
        clf = MultinomialNB()
        clf.fit(bigram_cnt_vectorizer_training, self.training_data.target)
        y_predict = clf.predict(bigram_cnt_vectorizer_test)
        precision, recall, f1 = self.calculate_scores(y_predict)
        result_data.append(['NB', 'BB', precision, recall, f1])

        # Logistic Regression for unigram
        clf = LogisticRegression()
        clf.fit(unigram_cnt_vectorizer_training, self.training_data.target)
        y_predict = clf.predict(unigram_cnt_vectorizer_test)
        precision, recall, f1 = self.calculate_scores(y_predict)
        result_data.append(['LR', 'UB', precision, recall, f1])

        # Logistic Regression for bigram
        clf = LogisticRegression()
        clf.fit(bigram_cnt_vectorizer_training, self.training_data.target)
        y_predict = clf.predict(bigram_cnt_vectorizer_test)
        precision, recall, f1 = self.calculate_scores(y_predict)
        result_data.append(['LR', 'BB', precision, recall, f1])

        # SVM for unigram
        clf = LinearSVC()
        clf.fit(unigram_cnt_vectorizer_training, self.training_data.target)
        y_predict = clf.predict(unigram_cnt_vectorizer_test)
        precision, recall, f1 = self.calculate_scores(y_predict)
        result_data.append(['SVM', 'UB', precision, recall, f1])

        # SVM for bigram
        clf = LinearSVC()
        clf.fit(bigram_cnt_vectorizer_training, self.training_data.target)
        y_predict = clf.predict(bigram_cnt_vectorizer_test)
        precision, recall, f1 = self.calculate_scores(y_predict)
        result_data.append(['SVM', 'BB', precision, recall, f1])

        # Random Forest for unigram
        clf = RandomForestClassifier()
        clf.fit(unigram_cnt_vectorizer_training, self.training_data.target)
        y_predict = clf.predict(unigram_cnt_vectorizer_test)
        precision, recall, f1 = self.calculate_scores(y_predict)
        result_data.append(['RF', 'UB', precision, recall, f1])

        # Random Forest for bigram
        clf = RandomForestClassifier()
        clf.fit(bigram_cnt_vectorizer_training, self.training_data.target)
        y_predict = clf.predict(bigram_cnt_vectorizer_test)
        precision, recall, f1 = self.calculate_scores(y_predict)
        result_data.append(['RF', 'BB', precision, recall, f1])

        return result_data

    def plot_f1_values(self):
        input_sizes = []
        plot_values = {
            'NB': [],
            'LR': [],
            'SVM': [],
            'RF': []
        }
        for i in range(int(len(self.training_data.data) / 100)):
            input_sizes.append(100 * (i+1))
            cnt_vectorizer_training, cnt_vectorizer_test = \
                self.vectorize_by_cnt(training_data=self.training_data.data[0:100 * (i+1)])

            # Multinomial NB for unigram
            clf = MultinomialNB()
            clf.fit(cnt_vectorizer_training, self.training_data.target[0:100 * (i+1)])
            y_predict = clf.predict(cnt_vectorizer_test)
            plot_values['NB'].append(f1_score(self.test_data.target, y_predict, average="macro"))

            # Logistic Regression for unigram
            clf = LogisticRegression()
            clf.fit(cnt_vectorizer_training, self.training_data.target[0:100 * (i+1)])
            y_predict = clf.predict(cnt_vectorizer_test)
            plot_values['LR'].append(f1_score(self.test_data.target, y_predict, average="macro"))

            # SVM for unigram
            clf = LinearSVC()
            clf.fit(cnt_vectorizer_training, self.training_data.target[0:100 * (i+1)])
            y_predict = clf.predict(cnt_vectorizer_test)
            plot_values['SVM'].append(f1_score(self.test_data.target, y_predict, average="macro"))

            # Random Forest for unigram
            clf = RandomForestClassifier()
            clf.fit(cnt_vectorizer_training, self.training_data.target[0:100 * (i+1)])
            y_predict = clf.predict(cnt_vectorizer_test)
            plot_values['RF'].append(f1_score(self.test_data.target, y_predict, average="macro"))

        plt.figure()
        plt.title("Learning Curve")
        plt.xlabel("Training data size")
        plt.ylabel("F1 Score")
        plt.grid()
        i = 0
        colors = ['k', 'b', 'g', 'r']
        for key, val in plot_values.items():
            plt.plot(input_sizes, val, color=colors[i], label=key)
            i += 1
        plt.legend()
        plt.show()


def remove_headers(given_data):
    result = []
    # For each given sample, split with blank
    # line and remove the first part
    for content in given_data.data:
        result.append(' '.join(content.split('\n\n')[1:]))

    return result


def run():
    if len(sys.argv) == 5:
        training_datapath = sys.argv[1]
        test_datapath = sys.argv[2]
        out_file = sys.argv[3]
        display = int(sys.argv[4])
    else:
        print("Invalid Input, please try again")
        return

    # Load files
    training_data = load_files(training_datapath, encoding='latin1')
    test_data = load_files(test_datapath, encoding='latin1')

    # Remove headers in both data
    training_data.data = remove_headers(training_data)
    test_data.data = remove_headers(test_data)

    target = ClassifyText(training_data, test_data)
    res = target.explore_classifiers()
    output = ''
    for val in res:
        output += ', '.join(list(map(str, val))) + '\n'

    with open(out_file, 'w') as f:
        f.write(output)

    if display:
        target.plot_f1_values()

if __name__ == '__main__':
    run()
