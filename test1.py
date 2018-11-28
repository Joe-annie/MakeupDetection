import cv2
import dlib
import os
import numpy as np
# import sklearn.model_selection


def read_image(img_name):
    im = cv2.imread(img_name)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    data = np.array(im) #(128, 128)
    return data 

# images = []
# labels = []
# img_path0 = './makeup_with_labels/no_makeup'
# for fn in os.listdir(img_path0):
#     if fn.endswith('.jpg'):
#         fd = os.path.join(img_path0,fn)
#         images.append(read_image(fd))
#         labels.append('0')
# img_path1 = './makeup_with_labels/yes_makeup'
# for fn in os.listdir(img_path1):
#     if fn.endswith('.jpg'):
#         fd = os.path.join(img_path1,fn)
#         images.append(read_image(fd))
#         labels.append('1')
# print('load success!')



# dlib预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
img_path0 = './makeup_with_labels/no_makeup'
path_save0 = './makeup_with_labels/no_makeup_face'
for fn in os.listdir(img_path0):
    if fn.endswith('.jpg'):
        #read img
        fd = os.path.join(img_path0,fn)
        img = read_image(fd)
        
        #获取图片的宽高
        img_shape=img.shape
        img_height=img_shape[0]
        img_width=img_shape[1]
        # dlib检测
        d = detector(img,1)
        if len(d) > 1: continue
        
        # 计算矩形大小
        pos_start = tuple([d.left(), d.top()])
        pos_end = tuple([d.right(), d.bottom()])
        # 计算矩形框大小
        height = d.bottom()-d.top()
        width = d.right()-d.left()
        # 根据人脸大小生成空的图像
        img_blank = np.zeros((height, width, 3), np.uint8)

        for i in range(height):
                for j in range(width):
                        img_blank[i][j] = img[d.top()+i][d.left()+j]
        # 存在本地
        filename = fn + '_face'
        print("Save to:", filename)
        cv2.imwrite(path_save0+"/"+filename, img_blank)














# X = np.array(images)
# Y = np.array(labels)
# print (X.shape) #(2479, 128, 128)
# print(Y.shape)  #(2479,)

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state= 30)

##### Classifier:

#SVM, sklearn 

# from sklearn.svm import LinearSVC
# model = LinearSVC()
# model.fit(X_train, y_train)#
# y_pred = model.predict(X_test)
# #evaluate
# print(classification_report(Y_test, Y_predict, target_names=digits.target_names.astype(str))) 


# CNN2, keras #导入包后会意外退出
# np.random.seed(1337)
# from keras.datasets import mnist
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers import Dropout, Dense, Activation, Convolution2D, MaxPooling2D, Flatten
# from keras.optimizers import Adam

# X_train = X_train.reshape(2479, 128, 128)/255
# X_test = X_test.reshape(2479, 128, 128)/255

# y_train = np_utils.to_categorical(y_train, 2)
# y_test = np_utils.to_categorical(y_test, 2)
############################################################
# model = Sequential()
# #conv layer 1 as follows
# model.add(Convolution2D(
# 		nb_filter = 64,
# 		nb_row = 5,
# 		nb_col = 5,
# 		border_mode = 'same',
# 		input_shape=(1,60,160)
# 						)
# 		)
# model.add(Activation('relu'))
# model.add(Dropout(0.2)) 

# #pooling layer 1 as follows
# model.add(MaxPooling2D(
# 					pool_size = (2,2),
# 					strides = (2,2),
# 					border_mode = 'same',
# 					)
# 		)

# #conv layer 2 as follows
# model.add(Convolution2D(128, 5, 5, border_mode = 'same'))
# model.add(Activation('relu'))
# model.add(Dropout(0.2)) 

# #pooling layer 2 as follows
# model.add(MaxPooling2D(2, 2, border_mode = 'same'))

# ########################
# model.add(Flatten())
# model.add(Dense(1024))
# model.add(Activation('relu'))

# ########################
# model.add(Dense(26))
# model.add(Activation('softmax'))

# ########################
# adam = Adam(lr = 1e-4)

# ########################
# model.compile(optimizer=adam,
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# print('Training ------------')
# # Another way to train the model
# model.fit(X_train, y_train, epochs=30, batch_size=64,)

# print('\nTesting ------------')
# # Evaluate the model with the metrics we defined earlier
# loss, accuracy = model.evaluate(X_test, y_test)

# print('\ntest loss: ', loss)
# print('\ntest accuracy: ', accuracy)


