from keras import models
import numpy as np
import cv2 as cv
import json


def load_config(fp):
    with open(fp) as f:
        config = json.load(f)
        return config['ctg'], config['input_size']


def decode(preds, ctg):
    results = []
    for pred in preds:
        i = pred.argmax()
        result = ctg[i]
        results.append(result)
    return results


def preprocess(arr, input_size):
    # resize
    x = cv.resize(arr, input_size)
    # BGR 2 RGB
    x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
    x = np.expand_dims(x, 0).astype('float32')
    x /= 255
    return x


def main():
    ctg, input_size = load_config('./config.json')
    input_size = tuple(input_size)
    model = models.load_model('mymodel.h5')
    cap = cv.VideoCapture(0)
    while True:
        _, f = cap.read()
        # predict
        x = preprocess(f, input_size)
        y = model.predict(x)
        r = decode(y, ctg)
        # plot result
        cv.putText(
            img=f,
            text=r[0],
            org=(250, 50),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 98, 255),
            thickness=2)
        # show image
        cv.imshow('webcam', f)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
