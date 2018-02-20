import glob
import os
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def is_letters(text):
    return text.isalpha()

def filter_text(files):
    cleaned_files = []
    for file in files:
        cleaned_files.append(' '.join([lemmatizer.lemmatize(word.lower())
                                       for word in file.split() if is_letters(word) and word not in people_names]))
    return cleaned_files

def get_category_index(categories):
    from collections import defaultdict
    category_index = defaultdict(list)
    for index, category in enumerate(categories):
        category_index[category].append(index)
    return category_index

def calculate_p_y(category_index):
    p_y = {category: len(index) for category,index in category_index.items() }
    total_count = sum(p_y.values())
    for category in p_y:
        p_y[category] /= float(total_count)
    return p_y

def get_p_xi_y(document_vector,category_index,smoothing):
    p_xi_y = {}
    for category,index in category_index.items():
        p_xi_y[category] = document_vector[index,:].sum(axis=0) + smoothing
        p_xi_y[category] = np.asarray(p_xi_y[category])[0]
        total_count = p_xi_y[category].sum()
        p_xi_y[category] = p_xi_y[category] / float(total_count)
    return p_xi_y

def get_p_y_xi(doc_matrix,p_y,p_xi_y):
    number_of_docs = doc_matrix.shape[0]
    p_y_xis = []
    for i in range(number_of_docs):
        p_y_xi = {key: np.log(p_y_category) for key,p_y_category in p_y.items()}
        for category, p_xi_y_category in p_xi_y.items():
            doc_vector = doc_matrix.getrow(i)
            counts = doc_vector.data
            indices = doc_vector.indices
            for count, index in zip(counts, indices):
                p_y_xi[category] += np.log(p_xi_y_category[index]) * count
        min_log_p_y_xi = min(p_y_xi.values())
        for label in p_y_xi:
            try:
                p_y_xi[label] = np.exp(p_y_xi[label] - min_log_p_y_xi)
            except:
                p_y_xi[label] = float('inf')
        sum_p_y_xi = sum(p_y_xi.values())
        for label in p_y_xi:
            if p_y_xi[label] == float('inf'):
                p_y_xi[label] = 1.0
            else:
                p_y_xi[label] /= sum_p_y_xi
        p_y_xis.append(p_y_xi.copy())
    return p_y_xis

files,categories = [], []
file_paths = ["train/20_newsgroups/"+d+"/" for d in os.listdir('train/20_newsgroups') if os.path.isdir(os.path.join('train/20_newsgroups', d))]

for file_path in file_paths:
    for file_name in glob.glob(os.path.join(file_path, '*')):
        with open(file_name, 'r', encoding = "ISO-8859-1") as infile:
            files.append(infile.read())
            categories.append(file_paths.index(file_path))

print("Training...")

people_names = set(names.words())
lemmatizer = WordNetLemmatizer()

cleaned_messages = filter_text(files)

cv = CountVectorizer(stop_words="english", max_features=500)
message_vectors = cv.fit_transform(cleaned_messages)

category_index = get_category_index(categories)
p_y = calculate_p_y(category_index)

smoothing = 100
p_xi_y = get_p_xi_y(message_vectors,category_index,smoothing)

print("Testing...")

test_files,test_categories = [], []
test_file_paths = ["test/20_newsgroups/"+d+"/" for d in os.listdir('test/20_newsgroups') if os.path.isdir(os.path.join('train/20_newsgroups', d))]

for file_path in test_file_paths:
    for file_name in glob.glob(os.path.join(file_path, '*')):
        with open(file_name, 'r', encoding = "ISO-8859-1") as infile:
            test_files.append(infile.read())
            test_categories.append(test_file_paths.index(file_path))

cleaned_messages_test = filter_text(test_files)
test_message_vectors = cv.fit_transform(cleaned_messages_test)
posterior = get_p_y_xi(test_message_vectors, p_y, p_xi_y)

prediction = []
prediction_np = np.zeros(len(posterior))

count=0
for post in posterior:
    prediction.append(np.argmax(np.asarray(list(post.values()))))
    prediction_np[count] = np.argmax(np.asarray(list(post.values())))
    count+=1

test_np = np.asarray(test_categories)

error = prediction_np == test_np
print("Accuracy is ",(error.sum() / error.shape[0])*100," percent.")
