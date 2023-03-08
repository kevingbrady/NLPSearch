import gensim
import os
from gensim import corpora
from ArxivDataManager.ChunkReader import ChunkReader
import fasttext.util as fasttext_util
from gensim.models import fasttext

if __name__ == '__main__':

    reader = ChunkReader('arxiv_processed/arxiv-2022-01-15 10:53:24.938899.csv')
    #dictionary = corpora.Dictionary()
    chunk = reader.getChunk()
    dictionary = corpora.Dictionary.load('ARXIV_Dictionary')
    bow_documents = []

    while chunk is not None:
        chunk['keywords'] = chunk['keywords'].apply(lambda x: x.strip('[]').split(', '))
        #dictionary.add_documents(chunk['keywords'].values)
        bow_documents += ([dictionary.doc2bow(x) for x in chunk['keywords']])
        chunk = reader.getChunk()

    if os.path.exists('FAST_TEXT_MODEL'):
        print('Loading FastText model from file ...')
        fasttext_model300 = fasttext.FastText.load('FAST_TEXT_MODEL')
        print('FastText Model Loaded')
    else:
        print('Downloading and Loading FastText Model ...')
        #fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
        fasttext_util.download_model('en', if_exists='ignore')
        fasttext_model300 = fasttext.load_facebook_model('cc.en.300.bin')
        print('FastText Model Loaded')
        fasttext_model300.save('FAST_TEXT_MODEL')
        print('FastText Model Loaded')

    tfidf = gensim.models.TfidfModel(dictionary=dictionary)

    if os.path.exists('similarity_matrix_fasttext'):
        print('Loading SimilarityMatrix from File ...')
        similarity_matrix = gensim.similarities.SparseTermSimilarityMatrix.load('similarity_matrix_fasttext')
        print('Similarity Matrix Loaded')

    else:
        print('Creating Similarity Index ...')
        similarity_index = gensim.similarities.WordEmbeddingSimilarityIndex(fasttext_model300.wv)
        print('Creating Similarity Matrix ...')
        similarity_matrix = gensim.similarities.SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf=tfidf)
        similarity_matrix.save('similarity_matrix_fasttext')
        print('Similarity Matrix Built')

    '''
    if os.path.exists('softcossim_matrix_fasttext'):
        print('Loading Soft Cosine Similarity Matrix from File ...')
        soft_matrix_fasttext = gensim.similarities.SoftCosineSimilarity.load('softcossim_matrix_fasttext')
        print('Soft Cose Similarity Matrix Loaded')
    else:
        print('Creating Soft Cosine Similarity Matrix ...')
        soft_matrix_fasttext = gensim.similarities.SoftCosineSimilarity(tfidf[bow_documents], similarity_matrix)
        soft_matrix_fasttext.save('softcossim_matrix_fasttext')
        print('Soft Cosine Similarity Matrix Built')
        '''
