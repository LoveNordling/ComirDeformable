import os
import sys
import csv
import subprocess
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk


def register_deformed_comirs(modA_path, modB_path, modA_comir_path, modB_comir_path, config_path, out_path):
    
    print("registering comirs")

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
        landmarkA_path = os.path.join(modA_path, landmark_filename)
        landmarkB_path = os.path.join(modB_path, landmark_filename)
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


def register_elastix(modA_path, modB_path, elastix_output_dir, gridspacing):
    print("registering elastix")
    if not os.path.exists(elastix_out_dir):
        os.makedirs(elastix_out_dir)
    running_mse_before = []
    running_mse_after = []

    filenames = os.listdir(modA_path)
    filenames_images = [x for x in filenames if  x.endswith(".tif") or x.endswith(".png")]
    filenames_landmarks = [x for x in filenames if x.endswith(".csv")]
    filenames_images.sort()
    filenames_landmarks.sort()
    N = len(filenames_images)
    for i, names in enumerate(zip(filenames_images, filenames_landmarks)):
        (image_name, landmarks_name) = names
        print("Registering image  {} {}/{}".format(image_name, i+1,N))
        landmarkA_path = os.path.join(modA_path, landmarks_name)
        landmarkB_path = os.path.join(modB_path, landmarks_name)
        
        pathA = os.path.join(modA_path, image_name)
        pathB = os.path.join(modB_path, image_name)

        registered_landmarks_path = os.path.join(elastix_out_dir, landmarks_name)

        registered_path = os.path.join(elastix_out_dir, image_name)
        
        fixedImage = sitk.ReadImage(pathB, sitk.sitkInt8)
        movingImage = sitk.ReadImage(pathA, sitk.sitkInt8)
        fixedImage = sitk.GetArrayFromImage(fixedImage)
        if len(fixedImage.shape) == 3:
            fixedImage = cv2.cvtColor(fixedImage, cv2.COLOR_BGR2GRAY)
        fixedImage = sitk.GetImageFromArray(fixedImage)


        movingImage = sitk.GetArrayFromImage(movingImage)
        if len(movingImage.shape) == 3:
            movingImage = cv2.cvtColor(movingImage, cv2.COLOR_BGR2GRAY)
        movingImage = sitk.GetImageFromArray(movingImage)
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixedImage)
        elastixImageFilter.SetMovingImage(movingImage)
        parameterMapVector = sitk.VectorOfParameterMap()
        parameterMap = sitk.GetDefaultParameterMap("bspline")

        spacings = ('3.5', '2.803221', '1.988100', '1.410000', '1.000000')
        n_res = len(spacings)
        parameterMap['NumberOfResolutions'] = [str(n_res)]
        parameterMap['GridSpacingSchedule'] = spacings[(len(spacings)-n_res)::]
        parameterMap['MaxumNumberOfIterations'] = ['1024']
        parameterMap['FinalGridSpacingInPhysicalUnits']=[str(gridspacing)]
        
        parameterMapVector.append(parameterMap)
        elastixImageFilter.SetParameterMap(parameterMapVector)

        elastixImageFilter.LogToConsoleOff()
        elastixImageFilter.Execute()

        transformParameterMap = elastixImageFilter.GetTransformParameterMap()

        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(transformParameterMap)
        transformixImageFilter.ComputeDeformationFieldOn()
        transformixImageFilter.LogToConsoleOff()
        transformixImageFilter.Execute()
        deformationField = transformixImageFilter.GetDeformationField()
        deformationField = sitk.GetArrayFromImage(deformationField)
        deformationField = deformationField.astype(np.float64)
        deformationField = sitk.GetImageFromArray(deformationField, True)

        transform = sitk.DisplacementFieldTransform(deformationField)
        pointsA = np.genfromtxt(landmarkA_path, delimiter=',')
        pointsB = np.genfromtxt(landmarkB_path, delimiter=',')

        registeredPointsB = []
        for i in range(pointsB.shape[0]):
            p = pointsB[i,:]
            newp = transform.TransformPoint(p)#cv2.KeyPoint(p1, p2))
            registeredPointsB.append(newp)

        registeredPointsB = np.array(registeredPointsB)
        
        pd.DataFrame(registeredPointsB).to_csv(registered_landmarks_path,index=False,header=False)
        resultsImage = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage()).astype(np.uint8)
        cv2.imwrite(registered_path, resultsImage)
        cv2.waitKey(0)
        
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
        """
        mse_before = np.abs(landmarksA-landmarksB).mean()
        mse_after = np.abs(landmarksA_registered - landmarksB).mean()
        """
        
        mse_before = np.sqrt(np.sum((landmarksA-landmarksB)**2, axis=-1))
        mse_after = np.sqrt(np.sum((landmarksA_registered-landmarksB)**2, axis=-1))
        
        #mse_after = np.abs(landmarksA_registered - landmarksB)
        running_mse_before.append(mse_before)
        running_mse_after.append(mse_after)
        
        
        #print("MSE A & B: {} --- MSE registered A & B: {}".format(mse_before, mse_after))
        

    #running_mse_before = [item for sublist in running_mse_before for item in sublist]
    #running_mse_after = [item for sublist in running_mse_after for item in sublist]
    running_mse_before = [x.mean() for x in running_mse_before]
    running_mse_after = [x.mean() for x in running_mse_after]
    #running_mse_before = [np.amax(x) for x in running_mse_before]
    #running_mse_after = [np.amax(x) for x in running_mse_after]
    
    running_mse_before = np.array(running_mse_before)
    running_mse_after = np.array(running_mse_after)
    
    accuracies_before = []
    accuracies_after = []
    thresholds = []
    threshold = 0
    step_size = 0.1
    

    print(np.mean(running_mse_after < running_mse_before) )
    
    for i in range(200):
        successes = np.sum(running_mse_after <= threshold)
        accuracy = successes/len(running_mse_after)
        accuracies_after.append(accuracy)
        thresholds.append(threshold)
        threshold = threshold + step_size
        
        successes = np.sum(running_mse_before <= threshold)
        accuracy = successes/len(running_mse_before)
        accuracies_before.append(accuracy)

    

    return thresholds, accuracies_before, accuracies_after






print(len(sys.argv), sys.argv)
if len(sys.argv) < 3:
    print('Use: inference_comir.py model_path mod_a_path mod_b_path mod_a_out_path mod_b_out_path')
    sys.exit(-1)

if __name__ == "__main__":

    root = sys.argv[1]
    config_path = sys.argv[2]
    modA_path = os.path.join(root, "A")
    modB_path = os.path.join(root, "B")
    modA_comir_path = os.path.join(root, "A_comir")
    modB_comir_path = os.path.join(root, "B_comir")
    out_path = os.path.join(root, "registered")
    elastix_out_dir = os.path.join(root, "elastix")
    gridspacing = sys.argv[3] #Good number is 16 for zuirch or 32 for eliceiri
    
    
    #register_deformed_comirs(modA_path, modB_path, modA_comir_path, modB_comir_path, config_path, out_path)
    #register_elastix(modA_path, modB_path, elastix_out_dir, gridspacing)
    
    thresholds, accuracies_before, accuracies_after = evaluate_registration(modA_path, modB_path, out_path)
    success_rate_noreg = accuracies_before
    success_rate_comir_inspire = accuracies_after
    
    plt.plot(thresholds, accuracies_before, label="No registration")
    plt.plot(thresholds, accuracies_after, label="INSPIRE registration")

    thresholds, accuracies_before, accuracies_after = evaluate_registration(modB_path, modA_path, elastix_out_dir)
    success_rate_elastix = accuracies_after
    no_regisration_result_path = os.path.join(out_path, "success_rate_no_registration.csv")
    comir_inspire_result_path = os.path.join(out_path, "success_rate_comir_inspire.csv")
    elastix_result_path = os.path.join(out_path, "success_rate_elastix.csv")
    pd.DataFrame(np.array([thresholds,success_rate_noreg]).T).to_csv(no_regisration_result_path,index=False,header=False)
    pd.DataFrame(np.array([thresholds,success_rate_comir_inspire]).T).to_csv(comir_inspire_result_path,index=False,header=False)
    pd.DataFrame(np.array([thresholds,success_rate_elastix]).T).to_csv(elastix_result_path,index=False,header=False)
    plt.plot(thresholds, accuracies_after, label="Elastix")
    
    plt.ylabel('Accuracy')
    plt.xlabel('Landmark distance threshold')
    
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(os.path.join(out_path, 'accuracies.png'))
    
    

    #print("Total landmark mse and median before registration: {}, {}".format(np.mean(running_mse_before), np.median(running_mse_before)))
    #print("Total landmark mse and median after registration: {}, {}".format(np.mean(running_mse_after), np.median(running_mse_after)))    
    
    
