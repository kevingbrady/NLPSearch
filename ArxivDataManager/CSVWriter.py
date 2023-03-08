

class csvWriter:

    filename = ''
    columns = ['id',
               'submitter',
               'authors',
               'title',
               'comments',
               'journal-ref',
               'doi',
               'report-no',
               'categories',
               'license',
               'abstract',
               'versions',
               'update_date',
               'authors_parsed',
               'checksum',
               'keywords'
               ]

    def writeChunk(self, chunk, count, start_time):
        header = True if count == 1 else False
        mode = 'w' if count == 1 else 'a'

        chunk.to_csv(
            '/Users/kevin/PycharmProjects/NLPSearch/arxiv_processed/arxiv-' + str(start_time) + '.csv',
            index=False, header=header,
            columns=self.columns, mode=mode)
