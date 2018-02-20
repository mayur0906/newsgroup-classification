#Imports
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

print("Training...")

#Newsgroup variables
newsgroup_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))

newsgroup_trainer = list()
for i in newsgroup_train.target_names:
    newsgroup_trainer.append(fetch_20newsgroups(subset='train',categories=[i],remove=('headers', 'footers', 'quotes')))

# Vectorizer initilization
cv = CountVectorizer(stop_words='english',max_features=9500)
cvs = list()

for i in range(20):
    cvs.append(CountVectorizer(stop_words='english',max_features=9500))

vec_newsgroup = cv.fit_transform(newsgroup_train.data)
newsgroup_vecs = list()

tok = RegexpTokenizer(r'\w+')
the_words = tok.tokenize(str(newsgroup_trainer[1].data))
# print(the_words)

j = cvs[1].fit_transform(newsgroup_trainer[1].data)

for a_cv in cvs:
    newsgroup_vecs.append(a_cv.fit_transform(newsgroup_trainer[cvs.index(a_cv)].data))

#P(Y)
p_y = np.ones(20)/20

y_count = list()
count=0
for i in range(20):
    y_count.append(dict(zip(cvs[i].get_feature_names(),np.asarray(newsgroup_vecs[i].sum(axis=0)).ravel())))

y_totals = list()
for newsgroup_vec in newsgroup_vecs:
    y_totals.append(len(newsgroup_vec.data))

whole_dict = dict(zip(cv.get_feature_names(),np.asarray(vec_newsgroup.sum(axis=0)).ravel()))
total_vocab = len(whole_dict)
accu1 = 85.09874627827

print("Testing...")

newsgroup_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

predictions = list()
tok = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
for curr_data in newsgroup_test.data:
#     vectorized_test = cv_test.fit_transform(curr_data)
    the_words = tok.tokenize(str(curr_data.lower()))
    the_words = [w for w in the_words if not w in stop_words]
    probs_y = np.zeros(20)
    for y in range(20):
        prod = 1/20
        for word in the_words:
            if word not in y_count[y]:
                prod *= ((1)/(total_vocab + y_totals[y]))
            else:
                prod *= ((1+y_count[y][word]) / (total_vocab + y_totals[y]))
        probs_y[y] = prod
    predictions.append(np.argmax(probs_y))
predictions_arr = np.asarray(predictions)
target_arr = np.asarray(newsgroup_test.target)

k = predictions_arr == target_arr
l = k.astype(int)
accu = (l.sum() / l.shape[0])*100
print("Accuracy is ",accu)
