from time import time
from datetime import datetime
from ChunkReader import ChunkReader
from ChunkHasher import ChunkHasher
from CSVWriter import csvWriter
import utils
import os
from gensim.corpora import Dictionary


class arXivUpdater:
    total_time = 0
    completed = 0

    def __init__(self, filename1, filename2):

        csv = csvWriter()

        reset = False
        if filename2 is None:
            reset = True

        else:
            arxiv_processed = ChunkReader(filename2)

        start_time = datetime.now()
        arxiv = ChunkReader(filename1)
        chunkOne = arxiv.getChunk()

        while chunkOne is not None:

            start = time()
            chunkTwo = None if reset else arxiv_processed.getChunk()

            # Hash each chunk to see if any updates need to be applied
            print('Processing Chunk %d ...' % arxiv.getCount())
            hasher = ChunkHasher()
            chunkOne, chunkTwo = hasher.hashChunks(chunkOne, chunkTwo)

            if hasher.checkEqual():
                print('Chunk %d hash matches arXiv hash, moving on ...' % arxiv.getCount())
                final_data = chunkTwo
            else:
                # Perform update on chunks to keep memory usage manageable
                print('Chunk %d hash does not match arXiv hash, extracting keywords ...' % arxiv.getCount())
                if chunkTwo is not None:
                    chunkOne['keywords'] = chunkTwo['keywords']

                chunkOne = chunkOne.apply(lambda x: utils.extract_keywords(x) if x.name in hasher.rowsToUpdate() else x, axis=1)
                final_data = chunkOne

            csv.writeChunk(final_data, arxiv.getCount(), start_time)
            #print(final_data)
            done = time()
            self.completed += len(final_data)
            print("Chunk %d [%d] completed in %.3f s" % (arxiv.getCount(), self.completed, done - start), end='\n\n')
            chunkOne = arxiv.getChunk()

        end_time = datetime.now()
        self.total_time = end_time - start_time
        print("File Loaded in %.2f seconds" % self.total_time.total_seconds())
