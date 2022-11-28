import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def conf_matrix(generator, model):
    y_pred = model.predict_generator(generator, generator.samples // generator.batch_size+1)
    y_pred = np.argmax(y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(generator.classes, y_pred))
    print('Classification Report')
    target_names = ['denuded', 'nondenuded']
    print(classification_report(generator.classes, y_pred, target_names=target_names))