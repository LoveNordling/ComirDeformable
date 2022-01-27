import SimpleITK as sitk
import numpy as np
import cv2


N = 834
grid_spacing = 64

reader = sitk.ImageFileReader()
reader.SetFileName("/data2/jiahao/Registration/Datasets/Eliceiri_patches/patch_tlevel3/A/test/1B_A1_R.tif")
image = reader.Execute();

array = sitk.GetArrayViewFromImage(image)


#def create_transform():
ctrl_pts = 7, 7
fix_edges = 2

ctrl_pts = np.array(ctrl_pts, np.uint32)
SPLINE_ORDER = 3
mesh_size = ctrl_pts - SPLINE_ORDER
transform = sitk.BSplineTransformInitializer(image, mesh_size.tolist())
params = transform.GetParameters()

grid_shape = *ctrl_pts, 2

max_displacement = 200
uv = np.random.rand(*grid_shape) - 0.5  # [-0.5, 0.5)
uv *= 2  # [-1, 1)

uv *= max_displacement


for i in range(fix_edges):
    uv[i, :] = 0
    uv[-1 - i, :] = 0
    uv[:, i] = 0
    uv[:, -1 - i] = 0

transform.SetParameters(uv.flatten(order='F').tolist())


resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(image)
resampler.SetTransform(transform)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(0.5)
resampler.SetOutputPixelType(sitk.sitkFloat32)
resampled = resampler.Execute(image)

array2 = sitk.GetArrayViewFromImage(resampled)
array3 = 1 - array2

print(np.max(array))
print(np.max(array2))

cv2.imshow("before", array/255)
cv2.imshow("after", array2/255)

cv2.waitKey(0)




