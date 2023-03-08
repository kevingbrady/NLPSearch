import os
import pandas as pd
from itertools import islice
from io import StringIO
from dataclasses import dataclass

@dataclass
class ChunkReader:

    file_path = ''
    file_name = ''
    file_type = ''
    chunk_generator = None
    CHUNK_SIZE = 100000
    count = 0

    def __init__(self, file):

        self.file_path = file
        self.file_name, self.file_type = os.path.splitext(file)

        self.readFile()

    def readFile(self):

        if self.file_type == '.json':
            self.chunk_generator = self.JSONChunkReader(self.file_path)

        if self.file_type == '.csv':
            self.chunk_generator = self.CSVChunkReader(self.file_path)

    def JSONChunkReader(self, file):

        with open(file) as filereader:
            while True:
                blockOne = list(islice(filereader, self.CHUNK_SIZE))

                if not blockOne:
                    # reached EOF
                    break
                chunk = pd.read_json(StringIO("[" + ",".join(blockOne) + "]"))
                yield chunk

    def CSVChunkReader(self, file):

        return pd.read_csv(file, chunksize=self.CHUNK_SIZE, iterator=True)

    def getChunk(self):

            def loadChunk(self):
                return next(self.chunk_generator, None)

            chunk = loadChunk(self)
            if chunk is not None:
                self.count += 1
                print('Loading Chunk %d from file %s ...' % (self.getCount(), self.file_name))
            return chunk

    def getCount(self):
        return self.count
