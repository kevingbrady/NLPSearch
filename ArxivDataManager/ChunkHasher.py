import pandas as pd
import numpy as np


class ChunkHasher:

    hash_check = []
    rows_to_update = []
    check = False

    def hashChunks(self, chunkOne, chunkTwo):

        chunkOne = chunkOne.replace({np.nan: '', 'NA': '', 'N/A': '', 'n/a': ''})
        chunkOne['checksum'] = pd.util.hash_pandas_object(
            chunkOne[['authors', 'comments', 'title', 'categories', 'abstract']], index=False)

        if chunkTwo is not None:
            if len(chunkOne) == len(chunkTwo):

                chunkTwo = chunkTwo.replace({np.nan: '', 'NA': '', 'N/A': '', 'n/a': ''})
                self.hash_check = chunkOne['checksum'].values == chunkTwo['checksum'].values
                self.rows_to_update = np.where(self.hash_check == False)[0]
                self.check = all(self.hash_check)

                return chunkOne, chunkTwo

        self.rows_to_update = chunkOne.index.values.tolist()
        return chunkOne, chunkTwo

    def rowsToUpdate(self):
        return self.rows_to_update

    def checkEqual(self):
        return self.check