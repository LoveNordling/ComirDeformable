import os
import sys
import csv
import subprocess
import numpy as np
import cv2
import matplotlib.pyplot as plt




def register_deformed_comirs(modA_path, modB_path, modA_comir_path, modB_comir_path, config_path, out_path):
    


    filenames = os.listdir(modA_path)
    filenames.sort()
    filenames = [x for x in filenames if x.endswith(".tif") or x.endswith(".png")]
    image_extension = filenames[0][-4:]
    
    filenames = [x.replace(".png", "").replace(".tif", "") for x in filenames]
    N = len(filenames)
    #comirnames = os.listdir(modA_comir_path)
    #comirnames = [x for x in comirnames if x.endswith(".tif") or x.endswith(".png")]
    if not os.path.exists(out_path):
        os.makedirs(out_path)

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


        #if i > 3:
        #    break

def evaluate_registration(pathA, pathB, registered_path):

    running_mse_before = []
    running_mse_after = []


    filenames = os.listdir(pathA)
    filenames.sort()
    filenames = [x for x in filenames if x.endswith(".csv")]
    N = len(filenames)
    for filename in filenames:
        landmarkA_path = os.path.join(pathA, filename)
        landmarkB_path = os.path.join(pathB, filename)
        landmark_registered_path = os.path.join(registered_path, filename)
        landmarksA = np.genfromtxt(landmarkA_path, delimiter=',')
        landmarksB = np.genfromtxt(landmarkB_path, delimiter=',')
        landmarksA_registered = np.genfromtxt(landmark_registered_path, delimiter=',')

        mse_before = ((landmarksA-landmarksB)**2).mean()
        mse_after = (np.abs((landmarksA_registered - landmarksB))).mean()
        running_mse_before.append(mse_before)
        running_mse_after.append(mse_after)
        print("MSE A and B: {} --- MSE registered A and B: {}".format(mse_before, mse_after))


    running_mse_before = np.array(running_mse_before)
    running_mse_after = np.array(running_mse_after)
    
    accuracies_before = []
    accuracies_after = []
    thresholds = []
    threshold = 0
    step_size = 0.1 
    for i in range(300):
        successes = np.sum(running_mse_after <= threshold)
        accuracy = successes/N
        accuracies_after.append(accuracy)
        thresholds.append(threshold)
        threshold = threshold + step_size
        
        successes = np.sum(running_mse_before <= threshold)
        accuracy = successes/N
        accuracies_before.append(accuracy)



    return thresholds, accuracies_before, accuracies_after


print(len(sys.argv), sys.argv)
if len(sys.argv) < 7:
    print('Use: inference_comir.py model_path mod_a_path mod_b_path mod_a_out_path mod_b_out_path')
    sys.exit(-1)

if __name__ == "__main__":

    modA_path = sys.argv[1]
    modB_path = sys.argv[2]
    modA_comir_path = sys.argv[3]
    modB_comir_path = sys.argv[4]
    config_path = sys.argv[5]
    out_path = sys.argv[6]
    
    register_deformed_comirs(modA_path, modB_path, modA_comir_path, modB_comir_path, config_path, out_path)
    
    thresholds, accuracies_before, accuracies_after = evaluate_registration(modA_comir_path, modB_comir_path, out_path)

    plt.plot(thresholds, accuracies_before, label="No registration")
    plt.plot(thresholds, accuracies_after, label="INSPIRE registration")
    plt.ylabel('Accuracy')
    plt.xlabel('Landmark distance threshold')
    
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(os.path.join(out_path, 'accuracies.png'))
    
    

    #print("Total landmark mse and median before registration: {}, {}".format(np.mean(running_mse_before), np.median(running_mse_before)))
    #print("Total landmark mse and median after registration: {}, {}".format(np.mean(running_mse_after), np.median(running_mse_after)))    
    
    
