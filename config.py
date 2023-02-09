#data_processing.py
PATH_TO_HILLSHADE = "C:/Users/pacoc/Desktop/Warsztat/Studia/mag/data/hillshade.tif"
PATH_TO_DENUDED = "C:/Users/pacoc/Desktop/Warsztat/Studia/mag/data/punkty/devided/denuded.gpkg"
PATH_TO_NONDENUDED = "C:/Users/pacoc/Desktop/Warsztat/Studia/mag/data/punkty/devided/nondenuded.gpkg"

#plots.py
PATH_TO_FMAPS = "plots/fmaps/"

#training.py
FMAP_SITES = ['denuded5', 'denuded13', 'denuded33', 'denuded34', 'nondenuded50', 'denuded51', 'nondenuded57', 'nondenuded58', 'nondenuded72', 'nondenuded89']

EPOCHS = 2
BATCH_SIZE = 20
IMG_WIDTH, IMG_HEIGHT = 128, 128