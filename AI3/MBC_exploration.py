import sys
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import precision_score, f1_score, recall_score


class ClassifyTextMBC(object):
    def __init__(self, training_data, test_data):
        self.training_data = training_data
        self.test_data = test_data
        # Create a single instance of Stemmer class
        self.ps = PorterStemmer()
        # Create stem data for training and test data sets for future use
        self.train_stem_data = list(map(self.stem_input, self.training_data.data))
        self.test_stem_data = list(map(self.stem_input, self.test_data.data))

    def vectorize_data(
            self,
            training_data=None,
            test_data=None,
            vectorizer_cls=CountVectorizer,
            lowercase=False,
            stop_words=None,
            unigram=True):

        if unigram:
            # Unigram baseline
            _min, _max = 1, 1
        else:
            # Bigram baseline
            _min, _max = 1, 2

        # Default to given training & test data if nothing is given
        if not training_data:
            training_data = self.training_data.data
        if not test_data:
            test_data = self.test_data.data

        # Vectorizer enforces lowercase words, stop words
        # filtering and unigram/bigram vectorisation
        data_vectorizer = vectorizer_cls(
            lowercase=lowercase,
            stop_words=stop_words,
            ngram_range=(_min, _max))
        # Fit and transform training data
        vectorizer_training = data_vectorizer.fit_transform(training_data)
        # Transform test data
        vectorizer_test = data_vectorizer.transform(test_data)
        return vectorizer_training, vectorizer_test

    def calculate_scores(self, y_predict):
        # Calculate macro-average scores given the predicted y value from classifier
        precision = precision_score(self.test_data.target, y_predict, average="macro")
        recall = recall_score(self.test_data.target, y_predict, average="macro")
        f1 = f1_score(self.test_data.target, y_predict, average="macro")
        return round(precision, 5), round(recall, 5), round(f1, 5)

    def stem_input(self, sentence):
        # Stemming given sentence using a tokenizer
        res = []
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(sentence)
        for word in words:
            res.append(self.ps.stem(word))

        return ' '.join(res)

    def multinomial_naive_bayes(self, vectorizer_training, vectorizer_test, alpha=1.0):
        # Predict using MNB classifier
        clf = MultinomialNB(alpha=alpha)
        # Use the vector created in the previous step
        clf.fit(vectorizer_training, self.training_data.target)
        # Predict for the test data vector
        y_predict = clf.predict(vectorizer_test)
        return y_predict

    def explore_features(self):
        # Buckets:
        # (Count & TFIDF Vectorizers with min_df, max_df)
        # (stop_words, lowercase, stemming - Porter & Snowball)
        # (hyperparameter alpha with different values)

        result_data = []

        # All cases are for unigram baseline
        # Multinomial NB with CountVectorizer + Lower case + Stemming
        cnt_vectorizer_training, cnt_vectorizer_test = self.vectorize_data(
            self.train_stem_data,
            self.test_stem_data,
            lowercase=True)
        y_predict = self.multinomial_naive_bayes(
            cnt_vectorizer_training,
            cnt_vectorizer_test)
        precision, recall, f1 = self.calculate_scores(y_predict)
        result_data.append(['NB', 'CountVectorizer + Lower case + Stemming',
                            precision, recall, f1])

        # Multinomial NB with CountVectorizer + Lower case + Stop_words
        cnt_vectorizer_training, cnt_vectorizer_test = self.vectorize_data(
            stop_words='english',
            lowercase=True)
        y_predict = self.multinomial_naive_bayes(
            cnt_vectorizer_training,
            cnt_vectorizer_test)
        precision, recall, f1 = self.calculate_scores(y_predict)
        result_data.append(['NB', 'CountVectorizer + Lower case + Stop_words',
                            precision, recall, f1])

        # Multinomial NB with CountVectorizer + Lower case + Stop_words + Alpha=0.01
        cnt_vectorizer_training, cnt_vectorizer_test = self.vectorize_data(
            stop_words='english',
            lowercase=True)
        y_predict = self.multinomial_naive_bayes(
            cnt_vectorizer_training,
            cnt_vectorizer_test,
            alpha=0.01)
        precision, recall, f1 = self.calculate_scores(y_predict)
        result_data.append(['NB', 'CountVectorizer + Lower case + Stop_words + Alpha=0.01',
                            precision, recall, f1])

        # Multinomial NB with CountVectorizer + Lower case + Stop_words + Alpha=0.001 + Stemming
        cnt_vectorizer_training, cnt_vectorizer_test = self.vectorize_data(
            self.train_stem_data,
            self.test_stem_data,
            stop_words='english',
            lowercase=True)
        y_predict = self.multinomial_naive_bayes(
            cnt_vectorizer_training,
            cnt_vectorizer_test,
            alpha=0.001)
        precision, recall, f1 = self.calculate_scores(y_predict)
        result_data.append(['NB', 'CountVectorizer + Lower case + Stop_words + Alpha=0.001 + Stemming',
                            precision, recall, f1])

        # Multinomial NB with TfidfVectorizer + Lower case + Stop_words + Stemming
        tfidf_vectorizer_training, tfidf_vectorizer_test = self.vectorize_data(
            self.train_stem_data,
            self.test_stem_data,
            vectorizer_cls=TfidfVectorizer,
            stop_words='english',
            lowercase=True)
        y_predict = self.multinomial_naive_bayes(
            tfidf_vectorizer_training,
            tfidf_vectorizer_test)
        precision, recall, f1 = self.calculate_scores(y_predict)
        result_data.append(['NB', 'TfidfVectorizer + Lower case + Stop_words + Stemming',
                            precision, recall, f1])

        # Multinomial NB with TfidfVectorizer + Lower case + Stemming + Alpha=0.1
        tfidf_vectorizer_training, tfidf_vectorizer_test = self.vectorize_data(
            self.train_stem_data,
            self.test_stem_data,
            vectorizer_cls=TfidfVectorizer,
            lowercase=True)
        y_predict = self.multinomial_naive_bayes(
            tfidf_vectorizer_training,
            tfidf_vectorizer_test,
            alpha=0.1)
        precision, recall, f1 = self.calculate_scores(y_predict)
        result_data.append(['NB', 'TfidfVectorizer + Lower case + Stemming + Alpha=0.1',
                            precision, recall, f1])

        # Multinomial NB with TfidfVectorizer + Lower case + Stop_words + Alpha=0.01
        tfidf_vectorizer_training, tfidf_vectorizer_test = self.vectorize_data(
            stop_words='english',
            vectorizer_cls=TfidfVectorizer,
            lowercase=True)
        y_predict = self.multinomial_naive_bayes(
            tfidf_vectorizer_training,
            tfidf_vectorizer_test,
            alpha=0.01)
        precision, recall, f1 = self.calculate_scores(y_predict)
        result_data.append(['NB', 'TfidfVectorizer + Lower case + Stop_words + Alpha=0.01',
                            precision, recall, f1])

        # Multinomial NB with TfidfVectorizer + Lower case + Stop_words + Alpha=0.001 + Stemming
        tfidf_vectorizer_training, tfidf_vectorizer_test = self.vectorize_data(
            self.train_stem_data,
            self.test_stem_data,
            stop_words='english',
            vectorizer_cls=TfidfVectorizer,
            lowercase=True)
        y_predict = self.multinomial_naive_bayes(
            tfidf_vectorizer_training,
            tfidf_vectorizer_test,
            alpha=0.001)
        precision, recall, f1 = self.calculate_scores(y_predict)
        result_data.append(['NB', 'TfidfVectorizer + Lower case + Stop_words + Alpha=0.001 + Stemming',
                            precision, recall, f1])

        return result_data


def remove_headers(given_data):
    result = []
    # For each given sample, split with blank
    # line and remove the first part
    for content in given_data.data:
        result.append(' '.join(content.split('\n\n')[1:]))

    return result


def run():
    if len(sys.argv) == 4:
        training_datapath = sys.argv[1]
        test_datapath = sys.argv[2]
        out_file = sys.argv[3]
    else:
        print("Invalid Input, please try again")
        return

    # Load files
    # TODO: check file input method
    training_data = load_files(training_datapath, encoding='latin1')
    test_data = load_files(test_datapath, encoding='latin1')

    # Remove headers in both data
    training_data.data = remove_headers(training_data)
    test_data.data = remove_headers(test_data)

    target = ClassifyTextMBC(training_data, test_data)
    res = target.explore_features()
    output = ''
    for val in res:
        output += ', '.join(list(map(str, val))) + '\n'

    with open(out_file, 'w') as f:
        f.write(output)

if __name__ == '__main__':
    run()
