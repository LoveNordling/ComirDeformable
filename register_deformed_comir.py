import os
import sys
import csv
import subprocess
import numpy as np
import cv2


print(len(sys.argv), sys.argv)
if len(sys.argv) < 7:
    print('Use: inference_comir.py model_path mod_a_path mod_b_path mod_a_out_path mod_b_out_path')
    sys.exit(-1)


modA_path = sys.argv[1]
modB_path = sys.argv[2]
modA_comir_path = sys.argv[3]
modB_comir_path = sys.argv[4]
config_path = sys.argv[5]
out_path = sys.argv[6]
    

filenames = os.listdir(modA_path)
filenames.sort()
filenames = [x for x in filenames if x.endswith(".tif") or x.endswith(".png")]
image_extension = filenames[0][-4:]

filenames = [x.replace(".png", "").replace(".tif", "") for x in filenames]
N = len(filenames)
#comirnames = os.listdir(modA_comir_path)
#comirnames = [x for x in comirnames if x.endswith(".tif") or x.endswith(".png")]

running_mse_before = []
running_mse_after = []
print(N)
for i, filename in enumerate(filenames):
    print("Registering image  {} {}/{}".format(filename, i+1,N))
    tforward_name = "tforward_" + filename + ".txt"
    treverse_name = "treverse_" + filename + ".txt"

    tforward_path = os.path.join(out_path, tforward_name)
    treverse_path = os.path.join(out_path, treverse_name)
    pathA = os.path.join(modA_path, filename + image_extension)
    pathB = os.path.join(modB_path, filename +  image_extension)

    comir_pathA = os.path.join(modA_comir_path, filename + image_extension)
    comir_pathB = os.path.join(modB_comir_path, filename + image_extension)
    landmark_filename = filename + ".csv"
    landmarkA_path = os.path.join(modA_comir_path, landmark_filename)
    landmarkB_path = os.path.join(modB_comir_path, landmark_filename)
    landmark_registered_path = os.path.join(out_path, landmark_filename)
    
    registered_path = os.path.join(out_path, filename + image_extension)
    comir_registered_path = os.path.join(out_path, "comir_" + filename + image_extension)
    

    #Perform registration
    process = subprocess.Popen(['../inspire-build/InspireRegister', '2', '-ref', comir_pathB, '-flo', comir_pathA, '-deform_cfg', config_path, '-out_path_deform_forward', tforward_path, '-out_path_deform_reverse', treverse_path],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    process = subprocess.Popen(['../inspire-build/InspireTransform', '-dim', '2', '-16bit', '1', 'interpolation', 'linear', '-transform', tforward_path, '-ref', comir_pathB, '-in', comir_pathA, '-out', comir_registered_path],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    process = subprocess.Popen(['../inspire-build/InspireTransform', '-dim', '2', '-16bit', '1', 'interpolation', 'linear', '-transform', tforward_path, '-ref', pathB, '-in', pathA, '-out', registered_path],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    process = subprocess.Popen(['../itkAlphaAMD-build/ACTransformLandmarks', '-dim', '2', '-transform', treverse_path, '-in', landmarkA_path, '-out', landmark_registered_path],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()


    landmarksA = np.genfromtxt(landmarkA_path, delimiter=',')
    landmarksB = np.genfromtxt(landmarkB_path, delimiter=',')
    landmarksA_registered = np.genfromtxt(landmark_registered_path, delimiter=',')

    mse_before = ((landmarksA-landmarksB)**2).mean()
    mse_after = ((landmarksA_registered - landmarksB)**2).mean()
    running_mse_before.append(mse_before)
    running_mse_after.append(mse_after)
    print(filename)
    print("MSE A and B: {} --- MSE registered A and B: {}".format(mse_before, mse_after))



print("Total landmark mse and median before registration: {}, {}".format(np.mean(running_mse_before), np.median(running_mse_before)))
print("Total landmark mse and median after registration: {}, {}".format(np.mean(running_mse_after), np.median(running_mse_after)))    
    
    
