import os
import tensorflow as tf
import time
import numpy as np
from ArxivDataManager.ChunkReader import ChunkReader
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse as sp
import pickle
import pandas as pd
import utils

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def preprocess(str):
    return str

@tf.function
def sparse_matrix_matmul(t1, t2):
    return tf.matmul(tf.sparse.to_dense(t1),
                     tf.sparse.to_dense(t2),
                     a_is_sparse=True,
                     b_is_sparse=True)


def sp_matrix_to_sp_tensor(x):
    """
    Converts a Scipy sparse matrix to a SparseTensor.
    :param x: a Scipy sparse matrix.
    :return: a SparseTensor.
    """
    if not hasattr(x, 'tocoo'):
        try:
            m = sp.coo_matrix(x)
        except:
            raise TypeError('x must be convertible to scipy.coo_matrix')
    else:
        m = x.tocoo()
    out = tf.SparseTensor(
        indices=np.array([m.row, m.col]).T,
        values=m.data,
        dense_shape=(m.shape[0], m.shape[1])
    )

    # BATCH_SIZE = 5.33e9 / (len(x.col) * 8)
    # print(BATCH_SIZE)
    return tf.sparse.reorder(out)


def build_sparse_matrix(file):
    reader = ChunkReader(file)
    chunk = reader.getChunk()
    documents = []

    while chunk is not None:
        # chunk['keywords'] = chunk['keywords'].apply(lambda x: x.strip('[]').split(', '))
        # chunk['keywords'] = chunk['keywords'].apply(lambda x: x.strip('[]').join(', '))
        # dictionary.add_documents(chunk['keywords'].values)
        # bow_documents += ([dictionary.doc2bow(x) for x in chunk['keywords']])

        chunk['keywords'] = chunk['keywords'].apply(lambda x: x.strip('[]').replace(',', ' '))
        documents += list(chunk['keywords'].values)
        chunk = reader.getChunk()

    print('Building TF-IDF BoW Matrix from %d entries ...' % (len(documents)))
    vectorizer = TfidfVectorizer(lowercase=False, preprocessor=preprocess, min_df=3, dtype=np.float32)
    tfidf_matrix = vectorizer.fit_transform(documents)
    features = vectorizer.get_feature_names_out()
    np.savetxt('/Users/kevin/PycharmProjects/NLPSearch/SimilarityMatrixBuilder/features.txt', features,
               fmt='%s',
               delimiter=',')
    with open("vectorizer.pk", 'wb') as file:
        pickle.dump(vectorizer, file)

    with open("tfidf_matrix.pk", 'wb') as file:
        pickle.dump(tfidf_matrix, file)

    print('TF-IDF BoW Matrix Built !!!')
    return vectorizer, tfidf_matrix


def print_run_time(elapsed_time):
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2} hrs {:0>2} min {:05.2f} s".format(int(hours), int(minutes), seconds)


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in tf.transpose(mat)]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")


def split_tensor_even_batches(tensor, num_entries, num_features, batch_size):
    print('Splitting Tensor into ', batch_size, " batches")
    split_size, remainder = divmod(num_entries, batch_size - 1)

    # Remove last batch so tensor can be reshaped to (batch_size, num_entries, num_features)
    last_batch = tf.sparse.slice(tensor, start=[split_size * (batch_size - 1), 0], size=[remainder, num_features])
    tensor = tf.sparse.slice(tensor, start=[0, 0], size=[split_size * (batch_size - 1), num_features])
    batches = tf.sparse.reshape(tensor, shape=((batch_size - 1), split_size, num_features))

    # Split reshaped tensor along (batch_size) dimesnion and then reshape again to effectively
    # split tensor into list of even chunks and the append the last batch
    batches = tf.sparse.split(batches, batch_size - 1, axis=0)
    batches = [tf.sparse.reshape(i, shape=(split_size, num_features)) for i in batches]
    batches.append(last_batch)

    # last_batch = tf.sparse.reshape(last_batch, shape=(1, remainder, num_features))
    # dataset = tf.data.Dataset.from_tensor_slices(batches)
    # last_batch = tf.data.Dataset.from_tensor_slices(last_batch)
    # dataset = dataset.concatenate(last_batch)

    return batches


if __name__ == '__main__':

    preprocessed_data = os.path.abspath('../arxiv_processed') + '/' + os.listdir('../arxiv_processed')[0]

    if os.path.exists("vectorizer.pk") and os.path.exists("tfidf_matrix.pk"):
        print("Loading TF-IDF BoW Matrix from file ...")
        with open("vectorizer.pk", "rb") as file:
            vectorizer = pickle.load(file)

        with open("tfidf_matrix.pk", "rb") as file:
            tfidf_matrix = pickle.load(file)
        print("TF-IDF BoW Matrix Loaded !!!")

    else:
        vectorizer, tfidf_matrix = build_sparse_matrix(preprocessed_data)

    wordlist = 'deep neural networks recurrent network security cryptography passive attacks'
    keywords = utils.generate_keywords(wordlist)

    tensor = sp_matrix_to_sp_tensor(tfidf_matrix)
    searchTensor = sp_matrix_to_sp_tensor(vectorizer.transform([" ".join(keywords)]))

    batch_size = 128
    num_entries = tensor.shape[0]
    num_features = tensor.shape[1]

    batches = split_tensor_even_batches(tensor, num_entries, num_features, batch_size)
    print('Starting Cosine Similarity Matrix Builder ...')

    start = time.time()

    '''
    for i, batchOne in enumerate(batches):
        print("batch %d of %d" % (i + 1, batch_size))
        for j, batchTwo in enumerate(batches):
            print('sub-batch %d of %d' % (j + 1, batch_size))

            similarity_matrix = sparse_matrix_matmul(batchOne, tf.sparse.transpose(batchTwo))

            # print(similarity_matrix.shape)
            # tf.keras.backend.clear_session()
        break
    '''
    idx = 0
    matrix = np.empty((num_entries,))

    for i, batchOne in enumerate(batches):
        output = sparse_matrix_matmul(searchTensor, tf.sparse.transpose(batchOne))
        arr = tf.squeeze(output).numpy()
        #print(idx, idx + (len(arr) - 1), matrix.shape)
        matrix.put([idx, (idx + (len(arr) - 1))], arr)
        idx += len(arr)

    end = time.time()
    indices = matrix.argsort()[-50:][::-1].tolist()
    #print(indices)
    dataset = pd.read_csv(preprocessed_data).iloc[indices]

    #with open("id_list.pk", "rb") as file:
    #    id_list = pickle.load(file)

    #output = [id_list[i] for i in indices]
    for idx, row in dataset.iterrows():
        print(row['categories'])
        print(row['title'])
        print()
        print(row['abstract'])
        print('\n\n')

    print('Completed in ' + print_run_time(end - start))
    # print('Projected Full Run Time ' + print_run_time(batch_size * (end - start)))
