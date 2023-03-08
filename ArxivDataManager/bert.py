import tensorflow as tf
import json
import time
import os
import pickle
from transformers import AutoTokenizer, TFAutoModel, BertModel
from transformers.onnx import FeaturesManager, export
from pathlib import Path
import onnx
from onnx_tf.backend import prepare
from optimum.onnxruntime import ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.configuration import OptimizationConfig
from optimum.pipelines import pipeline
from ChunkReader import ChunkReader

CHUNK_SIZE = 112000
BATCH_SIZE = 128
ELEMENTS_PER_BATCH = int(CHUNK_SIZE/BATCH_SIZE)
BERT_TOKEN_SIZE = 128


with open('categories.json') as f:
    categories = json.load(f)


def setup_model(bert_directory, model_name):
   # if os.path.exists(bert_directory + '/' + tokenizer_name + '.pkl'):
   #     tokenizer = pickle.load(open(bert_directory + '/' + tokenizer_name + '.pkl', 'rb'))
   # else:

   #     pickle.dump(tokenizer, open(bert_directory + '/' + tokenizer_name + '.pkl', 'wb'))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    onnx_path = Path("models/onnx/" + model_name + ".onnx")
    model = BertModel.from_pretrained(model_name)

   # load config

    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature='sequence-classification')
    onnx_config = model_onnx_config(model.config)

    # export
    export(preprocessor=tokenizer, model=model, config=onnx_config, opset=11, output=onnx_path)

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    '''
    if os.path.exists(bert_directory + '/' + model_name.replace('/', '_') + '.pkl'):
        model = pickle.load(open(bert_directory + '/' + model_name.replace('/', '_') + '.pkl', 'rb'))
    else:
        onnx_path = Path("onnx")
        model = ORTModel.fron_pretrained(model_name, from_transformers=True)
        pickle.dump(model, open(bert_directory + '/' + model_name.replace('/', '_') + '.pkl', 'wb'))
    '''
    return prepare(onnx_model, device='CUDA'), tokenizer


def get_category_keywords(text):

    category_keywords = []
    for category in text.split(' '):
        if category.__contains__('.'):
            main, sub = category.split('.')
            if main in categories:
                if sub in categories[main]:
                    category_keywords += categories[main]['desc']
                    category_keywords += categories[main][sub]
        else:
            if category in categories:
                category_keywords += categories[category]['desc']

    return category_keywords


@tf.function
def build_bert_sentence_vector(mask, embedding):

    print(mask.shape)
    print(embedding.shape)

    masked_embedding = embedding * mask
    #print(masked_embedding.shape)

    summed = tf.math.reduce_sum(masked_embedding, 1)
    #print(summed.shape)

    summed_mask = tf.clip_by_value(tf.math.reduce_sum(mask, 1), clip_value_min=1e-9, clip_value_max=1e9)

    mean_pooled = summed / summed_mask
    return mean_pooled


def convert_to_bert(row, tokenizer, input_ids, attention_mask, token_type_ids):

    title = row["title"]
    #comments = row["comments"]
    row_categories = row["categories"]
    abstract = row["abstract"]

    category_keywords = get_category_keywords(row_categories)

    input = title + abstract + " ".join(category_keywords)
    tokens = tokenizer.encode_plus(input,
                                   max_length=BERT_TOKEN_SIZE,
                                   truncation=True,
                                   return_attention_mask=True,
                                   add_special_tokens=True,
                                   padding='max_length',
                                   return_tensors='tf')

    input_ids.append(tokens["input_ids"][0])
    attention_mask.append(tokens["attention_mask"][0])
    #token_type_ids.append(tokens["token_type_ids"][0])

    return row


if __name__ == '__main__':

    arxiv = ChunkReader('../arxiv_metadata/arxiv-metadata-oai-snapshot.json')
    arxiv.CHUNK_SIZE = CHUNK_SIZE

    model_names = ['bert-base-uncased', 'distilbert-base-uncased', 'distilbert-base-uncased-finetuned-sst-2-english']
    onnx_model, tokenizer = setup_model('bert', model_names[1])

    chunk = arxiv.getChunk()

    while chunk is not None:
        input_ids = []
        attention_mask = []
        token_type_ids = []
        tokens = {}

        chunk_bert = chunk.apply(convert_to_bert, args=(tokenizer, input_ids, attention_mask, token_type_ids), axis=1)

        tokens["input_ids"] = tf.reshape(input_ids, shape=(BATCH_SIZE, ELEMENTS_PER_BATCH, BERT_TOKEN_SIZE))
        tokens["attention_mask"] = tf.reshape(attention_mask, shape=(BATCH_SIZE, ELEMENTS_PER_BATCH, BERT_TOKEN_SIZE))
        #tokens["token_type_ids"] = tf.reshape(token_type_ids, shape=(BATCH_SIZE, ELEMENTS_PER_BATCH, BERT_TOKEN_SIZE))

        bert_embeddings = []

        for x in range(0, BATCH_SIZE - 1):            #CHUNK_SIZE, ELEMENTS_PER_BATCH):

            start = time.time()
            print('(', x+1, '/', BATCH_SIZE, ')')

            X = tf.squeeze(tokens["input_ids"][x])
            Y = tf.squeeze(tokens["attention_mask"][x])
            #Z = tf.squeeze(tokens["token_type_ids"][x])

            output = onnx_model.run({"input_ids": X, "attention_mask": Y}) #, "token_type_ids": Z})

            mask = tf.cast(tf.tile(tf.expand_dims(Y, -1), [1, 1, output.logits.shape[-1]]), tf.float32)
            sentence_vectors = build_bert_sentence_vector(mask, output.logits)
            print(sentence_vectors.shape)

            end = time.time()
            print("%d s" % (end - start))

        exit(0)

        #chunk_bert["bert"] = build_bert_sentence_vector(mask, embedding)
        chunk = arxiv.getChunk()
