#data_processing.py
PATH_TO_HILLSHADE = "C:/Users/pacoc/Desktop/Warsztat/Studia/mag/data/hillshade.tif"
PATH_TO_DENUDED = "C:/Users/pacoc/Desktop/Warsztat/Studia/mag/data/punkty/devided/denuded.gpkg"
PATH_TO_NONDENUDED = "C:/Users/pacoc/Desktop/Warsztat/Studia/mag/data/punkty/devided/nondenuded.gpkg"

#plots.py
PATH_TO_FMAPS = "plots/fmaps/"

#training.py
FMAP_SITES = ['denuded/denuded5', 'denuded/denuded13', 'denuded/denuded33', 'denuded/denuded34', 'nondenuded/nondenuded50', 'denuded/denuded51', 'denuded/denuded57', 'nondenuded/nondenuded58', 'nondenuded/nondenuded72', 'nondenuded/nondenuded89']

EPOCHS = 250
BATCH_SIZE = 20
IMG_WIDTH, IMG_HEIGHT = 128, 128