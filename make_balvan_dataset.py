import skimage
import skimage.io as skio

import cv2
import csv

import os
import sys
import os
from tqdm import tqdm
import numpy as np


def prepare_filenames(rootA, rootB):
    filenamesA = os.listdir(rootA)
    filenamesA = [x.replace("BLUE_", "") for x in filenamesA]

    filenamesB = os.listdir(rootB)
    filenamesB = [x.replace("QPI_", "") for x in filenamesB]

    filenames = []
    for name in filenamesA:
        if name in filenamesB:
            filenames.append(name)

    filenamesA = ["BLUE_"+ x for x in filenames]
    filenamesB = ["QPI_"+ x for x in filenames]

    return filenamesA, filenamesB, filenames

def normalize(im, p):
    low = np.percentile(im, 100.0*p)
    hi = np.percentile(im, 100.0*(1.0-p))
    im = (im-low)/(hi-low+1e-15)
    im = np.clip(im, 0.0, 1.0)
    im = im * 255.0
    im = im.astype('uint8')
    return im



rootA = sys.argv[1]
rootB = sys.argv[2]
outpath = sys.argv[3]
outdirA = os.path.join(outpath, "A")
outdirB = os.path.join(outpath, "B")



filenamesA, filenamesB, filenames = prepare_filenames(rootA, rootB)
print(filenames)
print(len(filenamesA), len(filenamesB))

for filenameA, filenameB, filename in zip(filenamesA, filenamesB, filenames):
    
    pathA = os.path.join(rootA, filenameA)
    pathB = os.path.join(rootB, filenameB)
    
    print("Loading video A: " + filenameA)
    vidA = skio.imread(pathA)
    vidA = skimage.img_as_float(vidA)
    print("Loading video B: " + filenameB)
    vidB = skio.imread(pathB)
    vidB = skimage.img_as_float(vidB)

    for i in tqdm(range(vidA.shape[0])):
        if i % 5 == 0:
            imgA = vidA[i,:,:]
            #imgA = cv2.normalize(imgA, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            imgA = normalize(imgA, 0.001)
            imgB = vidB[i,:,:]
            #imgB = cv2.normalize(imgB, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            imgB = normalize(imgB, 0.0025)
                
            w,h = imgA.shape
            w = w//2
            h = h//2
        
            #imgA = imgA[w-150:w+150,h-150:h+150]
            #imgB = imgB[w-150:w+150,h-150:h+150]

        
            out_filename = filename.replace(".tif", "_" + f'{i:03d}' + ".tif")
            outpathA = os.path.join(outdirA, out_filename)
            outpathB = os.path.join(outdirB, out_filename)
            #print(outpathA)

        
            
            cv2.imwrite(outpathA, imgA)
            cv2.imwrite(outpathB, imgB)
            #cv2.imshow("image A", imgA)
            #cv2.imshow("image B", imgB)
            #cv2.waitKey(0)

    


    
