import os
from kaggle.api.kaggle_api_extended import KaggleApi
import logging
import time
from ArxivDataManager.ArxivUpdater import arXivUpdater
import glob
import utils

if __name__ == '__main__':

    logging.basicConfig(filename='../NLPSearch.log', filemode='w', level=logging.DEBUG,
                        format='%(asctime)s %(message)s')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    startTime = time.time()

    api = KaggleApi()
    api.authenticate()

    logging.info('Downloading arXiv metadata JSON file from Kaggle ...')
    api.dataset_download_files(dataset='Cornell-University/arxiv', path='/', unzip=True)
    logging.info('Metadata downloaded!!')

    if utils.check_arXiv_metdata("arxiv_metadata/arxiv-metadata-oai-snapshot.json") is False:
    #if True:
        logging.info(
            "Existing arxiv-metadata-oai-snapshot.json does not match the one downloaded from Kaggle, updates needed")

        logging.info("Loading Arxiv metadata from Kaggle into Pandas dataframe ...")
        arxiv = None
        if len(os.listdir("//arxiv_processed")) > 0:
            arxiv = glob.glob("//arxiv_processed/*.csv")[0]

        arXivUpdater("arxiv-metadata-oai-snapshot.json", arxiv)
        logging.info("arXiv update finished")

        if arxiv is not None:
            # Remove old arxiv csv file if the update is successful
            os.remove(glob.glob("//arxiv_processed/*.csv")[1])

        os.replace('/arxiv_metadata/arxiv-metadata-oai-snapshot.json',
                   '//arxiv_metadata/arxiv-metadata-oai-snapshot.json')

    else:
        logging.info('arXiv Metadata file has not changed since last processing run')
        if os.path.exists("/arxiv_metadata/arxiv-metadata-oai-snapshot.json"):
            os.remove("/arxiv_metadata/arxiv-metadata-oai-snapshot.json")



