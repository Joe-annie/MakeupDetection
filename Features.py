
import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter
import os
import cv2
from sklearn.model_selection import train_test_split
import mahotas as mt

def read_image(img_name):
    im = cv2.imread(img_name)
    #im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    data = np.array(im) #(128, 128)
    return data

images = []
labels = []
img_path0 = './makeup_with_labels/no_makeup'
for fn in os.listdir(img_path0):
    if fn.endswith('.jpg'):
        fd = os.path.join(img_path0,fn)
        images.append(read_image(fd))
        labels.append('0')
img_path1 = './makeup_with_labels/yes_makeup'
for fn in os.listdir(img_path1):
    if fn.endswith('.jpg'):
        fd = os.path.join(img_path1,fn)
        images.append(read_image(fd))
        labels.append('1')
print('load success!')

X = np.array(images)
Y = np.array(labels)
print (X.shape) #(2479, 128, 128,3)
print(Y.shape)  #(2479,)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state= 30)


# Subsample data

num_training=500
num_validation= 5
num_test= 400

mask = list(range(num_validation))
X_val = X_train[mask]
y_val = y_train[mask]

'''Detect ROI'''

face_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.6/site-packages/cv2/data/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.6/site-packages/cv2/data/Mouth.xml')

def detect_roi(X,y,face_flag=1):
    X_roi = np.zeros((X.shape[0],208))
    for index,img in enumerate(X):
        # if idx > 0:break # Test line
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.2, 3)
        if len(faces) < 1:  # no face detect,choose original img
            roi_gray = gray
            roi_color = img
        else:
            for (x, y, w, h) in faces:  # detect face
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
        img_face = roi_color
        if face_flag == 1:
            roi =  np.array(img_face,np.uint8)
        else:
            roi = np.zeros((52,1))
            # Eye
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 3)
            for idx, (ex, ey, ew, eh) in enumerate(eyes):
                if idx > 1 : break
                # crop roi img
                img_eye = roi_color[ey:ey + eh, ex:ex + ew]
                img_eye = np.array(img_eye)
                img_eye = cv2.resize(img_eye,(52,52),interpolation=cv2.INTER_CUBIC)
                # assert img_eye.shape[0] == 52, 'Feature  must be same dimensional'
                roi = np.hstack((roi,img_eye))
            roi = roi[:,1:]
            assert  roi.shape[1] == 104, 'detect error'
            # Mouth & Nose
            mouth = mouth_cascade.detectMultiScale(roi_gray, 1.2, 3)
            for idx, (mx, my, mw, mh) in enumerate(mouth):
                if idx > 1 : break
                # crop roi img
                img_mouth = roi_color[my:my + mh, mx:mx + mw]
                img_mouth = np.array(img_mouth)
                img_mouth = cv2.resize(img_mouth,(52,52),interpolation=cv2.INTER_CUBIC)
                # assert img_mouth.shape[0] == 52, 'Feature  must be same dimensional'
                roi = np.hstack((roi,img_mouth))
            assert roi.shape[1] == 208, 'detect error'
        print(roi.shape)
        X_roi[index] = roi
    return X_roi

X_val_roi = detect_roi(X_val,0) # return type is list
# X_train_roi = detect_roi(X_train,0) # return type is list
# X_test_roi = detect_roi(X_test,0) # return type is list

print(X_val_roi.shape)

'''Features funcitons:'''

def extract_features(imgs, feature_fns, verbose=False):
    """
    Given pixel data for images and several feature functions that can operate on
    single images, apply all feature functions to all images, concatenating the
    feature vectors for each image and storing the features for all images in
    a single matrix.

    Inputs:
    - imgs: N x H X W X C array of pixel data for N images.
    - feature_fns: List of k feature functions. The ith feature function should
      take as input an H x W x D array and return a (one-dimensional) array of
      length F_i.
    - verbose: Boolean; if true, print progress.

    Returns:
    An array of shape (N, F_1 + ... + F_k) where each column is the concatenation
    of all features for a single image.
    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # Use the first image to determine feature dimensions
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())  # shape中为1的维度去掉
        assert len(feats.shape) == 1, 'Feature  must be one-dimensional'
        feature_dims.append(feats.size)  # fn1 dim + fn2 dim +...
        first_image_features.append(feats)

    # Now that we know the dimensions of the features, we can allocate a single
    # big array to store all features as columns.
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))  # init feature matrix
    imgs_features[0] = np.hstack(first_image_features).T  # hstack之后为什么还要T？？？？？

    # Extract features for the rest of the images.
    for i in range(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):  # 用idx为标志把
            next_idx = idx + feature_dim
            imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        if verbose and i % 100 == 0:
            print('Done extracting features for %d / %d images' % (i, num_images))

    return imgs_features

def extract_features_list(imgs, feature_fns, verbose=False):
    """
    Given pixel data for images and several feature functions that can operate on
    single images, apply all feature functions to all images, concatenating the
    feature vectors for each image and storing the features for all images in
    a single matrix.

    Inputs:
    - imgs: N x (H52 X W?? X C3) list of pixel data for N images.
    - feature_fns: List of k feature functions. The ith feature function should
      take as input an H x W x D array and return a (one-dimensional) array of
      length F_i.
    - verbose: Boolean; if true, print progress.

    Returns:
    An array of shape (N, F_1 + ... + F_k) where each column is the concatenation
    of all features for a single image.
    """
    num_images = len(imgs)
    if num_images == 0:
        return np.array([])

    # Use the first image to determine feature dimensions
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0])  # shape中为1的维度去掉
        assert len(feats.shape) == 1, 'Feature  must be one-dimensional'
        first_image_features.append(feats)

    # Now that we know the dimensions of the features, we can allocate a single
    # big array to store all features as columns.
    imgs_features = list(range(num_images)) # init feature matrix
    imgs_features[0] = np.hstack(first_image_features).T  # hstack之后为什么还要T？？？？？

    # Extract features for the rest of the images.
    for i in range(1, num_images):
        l = []
        for feature_fn in feature_fns:  # 用idx为标志把
            l.append(feature_fn(imgs[i]))
        imgs_features[i] = l
        if verbose and i % 100 == 0:
            print('Done extracting features for %d / %d images' % (i, num_images))

    return imgs_features


# extract method 1:
def preprocessing(main_img):
    img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
    gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gs, (25, 25), 0)
    ret_otsu, im_bw_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((50, 50), np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
    return closing

def shape(img):
    closing = preprocessing(img)
    image, contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    M = cv2.moments(cnt)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    rectangularity = w * h / area
    circularity = ((perimeter) ** 2) / area
    return np.array([area,perimeter, w, h,aspect_ratio,rectangularity,circularity])

def color(main_img):
    img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
    red_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    blue_channel = img[:, :, 2]
    blue_channel[blue_channel == 255] = 0
    green_channel[green_channel == 255] = 0
    red_channel[red_channel == 255] = 0

    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)

    red_std = np.std(red_channel)
    green_std = np.std(green_channel)
    blue_std = np.std(blue_channel)
    return np.array([red_mean,green_mean,blue_mean,red_std,green_std,blue_std])

def texture(main_img):
    img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
    gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    textures = mt.features.haralick(gs)
    ht_mean = textures.mean(axis=0)
    contrast = ht_mean[1]
    correlation = ht_mean[2]
    inverse_diff_moments = ht_mean[4]
    entropy = ht_mean[8]
    return np.array([contrast,correlation,inverse_diff_moments,entropy])

# extract method 2：

def rgb2gray(rgb):
    """Convert RGB image to grayscale

      Parameters:
        rgb : RGB image

      Returns:
        gray : grayscale image

    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def hog_feature(im):
    """Compute Histogram of Gradient (HOG) feature for an image

         Modified from skimage.feature.hog
         http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog

       Reference:
         Histograms of Oriented Gradients for Human Detection
         Navneet Dalal and Bill Triggs, CVPR 2005

      Parameters:
        im : an input grayscale or rgb image

      Returns:
        feat: Histogram of Gradient (HOG) feature

    """

    # convert rgb to grayscale if needed
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.at_least_2d(im)

    sx, sy = image.shape  # image size
    orientations = 9  # number of gradient bins
    cx, cy = (8, 8)  # pixels per cell

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # compute gradient on x-direction
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # compute gradient on y-direction
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # gradient magnitude
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # gradient orientation

    n_cellsx = int(np.floor(sx / cx))  # number of cells in x
    n_cellsy = int(np.floor(sy / cy))  # number of cells in y
    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        # create new integral image for this orientation
        # isolate orientations in this range
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                            grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i,
                            temp_ori, 0)
        # select magnitudes for those orientations
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
    orientation_histogram[:, :, i] = uniform_filter(temp_mag, size=(cx, cy))[int(cx / 2)::cx, int(cy / 2)::cy].T

    return orientation_histogram.ravel()


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """
    Compute color histogram for an image using hue.

    Inputs:
    - im: H x W x C array of pixel data for an RGB image.
    - nbin: Number of histogram bins. (default: 10)
    - xmin: Minimum pixel value (default: 0)
    - xmax: Maximum pixel value (default: 255)
    - normalized: Whether to normalize the histogram (default: True)

    Returns:
      1D vector of length nbin giving the color histogram over the hue of the
      input image.
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    hsv = matplotlib.colors.rgb_to_hsv(im / xmax) * xmax
    imhist, bin_edges = np.histogram(hsv[:, :, 0], bins=bins, density=normalized)
    imhist = imhist * np.diff(bin_edges)

    # return histogram
    return imhist


# extract method 3:
import lib.Descriptors as des

def color_correlogram(img):
    # imgAutoCorr = cv2.imread(img, 1)
    imgAutoCorr = img
    matrix = des.autoCorrelogram(imgAutoCorr)
    # Klist = [1,3,5,7]
    # for idx,k in enumerate(Klist):
    #     print ("k = ", k)
    #     print (matrix[idx]) #[0.3521739130434783, 0.6478260869565218]
    return np.array(matrix[3])

def texture_lbp(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    transformed_img = des.lbp(img)
    # return np.array(transformed_img).ravel() ##16384
    return np.mean(np.array(transformed_img), axis=0, keepdims=True).ravel() #128

def shape_hog(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog = des.HoG(img,See_graph=True)
    return hog.ravel()


'''Extract Features'''

# feature_fns = [color,shape,texture] #17
#[color_correlogram,#2 texture_lbp #16384 ,hog_feature #2.3k] #18690
# feature_fns = [color_correlogram,texture_lbp,hog_feature]
feature_fns = [color_histogram_hsv,texture_lbp]


# X_train_feats = extract_features(X_train, feature_fns,verbose=True)
# X_val_feats = extract_features(X_val, feature_fns)
# X_test_feats = extract_features(X_test, feature_fns,verbose=True)

'存成list还是会出问题，用一些规则筛去检测器的结果长度不一的情况'
X_val_feats = extract_features(X_val_roi, feature_fns)
# X_train_feats = extract_features(X_train_roi, feature_fns)
# X_test_feats = extract_features(X_test_roi, feature_fns)


# print(len(X_val_feats))
# print(len(X_val_feats[0]))
print(X_val_feats.shape)


# Preprocessing: Subtract the mean feature
# mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
# X_train_feats -= mean_feat
# X_val_feats -= mean_feat
# X_test_feats -= mean_feat

''' Classifier:'''

# SVM, sklearn

# from sklearn.svm import LinearSVC
# from sklearn.metrics import classification_report
#
# model = LinearSVC()
# model.fit(X_train_feats, y_train)#
# y_pred = model.predict(X_test_feats)
# print(classification_report(y_test, y_pred))

# XGBoost, sklearn

# from xgboost.sklearn import XGBClassifier
# xgbc = XGBClassifier()
# xgbc.fit(X_train_feats,y_train)
# y_pred = xgbc.predict(X_test_feats)
# print(classification_report(y_test, y_pred))
