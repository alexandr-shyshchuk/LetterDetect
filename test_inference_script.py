from PIL import Image, ImageOps, ImageEnhance
from PIL.Image import Resampling
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from keras.models import load_model


def get_images(input_dir):
    imgs = []
    size = (28, 28)
    filenames = []

    if not os.path.exists(input_dir):
        print('wrong dir')
        return


    for filename in os.listdir(input_dir):
    
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            if filename.endswith(".png"):
                img = Image.open(os.path.join(input_dir, filename)).convert('RGBA')
                alpha = img.split()[-1]
                bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
                bg.paste(img, mask=alpha)
                img = bg.convert('RGB')
            else:
                img = Image.open(os.path.join(input_dir, filename)).convert('RGB')

            img = ImageEnhance.Color(img).enhance(0.0)
            img = img.convert('L').resize(size, resample=Resampling.LANCZOS)
            color = img.load()[0, 0]
        
            if color == 255:
                img = ImageOps.invert(img)

            imgs.append(img)
            filenames.append(input_dir + '/' +filename)
    
    data = np.empty((len(imgs), 28, 28))
    for i in range(len(imgs)):
        pixels = np.asarray(imgs[i], dtype=np.float32)
        pixels /= 255.0
        data[i] = np.asarray(pixels, dtype=np.float32)
    return data, filenames


def get_predicted_class(result):
    idx = np.argmax(result)
    if idx <= 9:
        return idx + 48
    elif 10 <= idx <= 35:
        return idx + 55
    else:
        return idx + 61
    return (idx)

def main():
    parser = argparse.ArgumentParser(description='Model')
    parser.add_argument('input_dir', metavar='input_directory', type=str, help='Directory with test images')

    args = parser.parse_args()

    input_dir = args.input_dir
    
    
    model = load_model("model.h5")
    images, filenames = get_images(input_dir)


    result = model.predict(images, verbose=0)
    
    for i in range(len(filenames)):
        idx = get_predicted_class(result[i])
        print(str(idx) + ", " + filenames[i])
    

    
if __name__ == '__main__':
    main()

