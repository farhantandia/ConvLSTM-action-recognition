import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, 
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras_video import VideoFrameGenerator,SlidingFrameGenerator
import glob
import matplotlib.pyplot as plt

# dataset
# classes = [i.split(os.path.sep)[1] for i in glob.glob('youtube data/normal/*')] #Youtube
classes = [i.split(os.path.sep)[1] for i in glob.glob('our data/30hz/*')] #our data
classes.sort()
print(len(classes))

img_height, img_width =100,100
SIZE = (img_height, img_width)
CHANNELS = 3
NBFRAME =20
Model_input_size = (NBFRAME, img_height, img_width, CHANNELS)
BS = 8
seq_len = NBFRAME

# data_dir_train = "youtube data/normal" #youtube
data_dir_train = "our_data/30hz"
 
# classes = ["Backstroke", "Breaststroke", "Freestyle","Safe","Drowning"]
'''MANUAL GENERATOR''' 
def frames_extraction(video_path, c, X, Y, Xf, Yf, sscnt,isTraining):
    frames_list = []
    flist = []
     
    vidObj = cv2.VideoCapture(video_path)
    # Used as counter variable 
    count = 1
    
    tmp_frames = []
    
    if isTraining:stride = 1
    else : stride =1
    while 1:
        success, image = vidObj.read()
        if success:
            if count % stride == 0:
                # image = image.astype(np.float32)
                # image /=255.0
                image = cv2.resize(image, (img_width, img_height))
#                 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                tmp_frames.append(image)
            count += 1
            if len(tmp_frames) == seq_len:
                sscnt += 1
                X.append(tmp_frames)
                
                y = [0]*len(classes)
                y[classes.index(c)] = 1
                Y.append(y)
                tmp_frames = []
                break
        else:
            #print("Defected frame")
            break
            
    return X, Y, Xf, Yf, sscnt
 
def create_data(input_dir,isTraining):
    X = []
    Y = []
    Xf = []
    Yf = []
    Xt = []
    Yt = []
    sscnt = 0
    for c in classes:
        print(c)
        if not (c in classes):
            continue
        files_list = os.listdir(os.path.join(input_dir, c))
        sscnt = 0
        for f in files_list:
            X, Y, Xf, Yf, sscnt = frames_extraction(os.path.join(os.path.join(input_dir, c), f), c, X, Y, Xf, Yf, sscnt,isTraining)
            
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y

X_train, y_train = create_data(data_dir_train,0)

# from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, shuffle=True, random_state=42)

print(y_train.shape)
# X_test, y_test = create_data(data_dir_train,0)

'''AUTO GENERATOR'''
# glob_pattern_train='project_data/{classname}/*.avi'
# glob_pattern_val='our_data/30hz/{classname}/*.avi'
# for data augmentation
# data_aug = ImageDataGenerator(
    # rescale=1./255.,
        # )
# # Create video frame generator
# train = VideoFrameGenerator(
#     classes=classes, 
#     glob_pattern=glob_pattern_train,
#     nb_frames=NBFRAME,
#     split_val=0.2,
#     shuffle=True,
#     batch_size=BS,
#     target_shape=SIZE,
#     nb_channel=CHANNELS,
#     transformation=data_aug,
#     use_frame_cache=True)

# train = SlidingFrameGenerator(
#     classes=classes, 
#     sequence_time=3,
#     glob_pattern=glob_pattern_train,
#     nb_frames=NBFRAME, 
#     split_val=.3, 
#     shuffle=True,
#     batch_size=BS,
#     target_shape=SIZE,
#     nb_channel=CHANNELS,
#     transformation=data_aug,
#     use_frame_cache=True)

# val = train.get_validation_generator()

# val=VideoFrameGenerator(
#     classes=classes, 
#     glob_pattern=glob_pattern_val,
#     nb_frames=NBFRAME,
#     shuffle=True,
#     batch_size=BS,
#     target_shape=SIZE,
#     nb_channel=CHANNELS,
#     transformation=data_aug,
#     use_frame_cache=True)
# valid = train.get_validation_generator()

filepath = 'models/convlstm_model.hdf5'
model = Sequential()

# model.add(Input(shape=Model_input_size))
# model.add(Conv3D(filters = 64, kernel_size = (3, 3, 3), padding = 'same', input_shape = Model_input_size))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# # model.add(Conv3D(filters = 64, kernel_size = (3, 3, 3), padding = 'same'))
# # model.add(BatchNormalization())
# # model.add(Dropout(0.5))

# model.add(Conv3D(filters = 128, kernel_size = (3, 3, 3), padding = 'same'))
# model.add(BatchNormalization())
# model.add(AveragePooling3D((2, 2, 2)))
# model.add(Dropout(0.5))

# model.add(Conv3D(filters = 256, kernel_size = (3, 3, 3), padding = 'same'))
# model.add(BatchNormalization())
# model.add(AveragePooling3D((2, 2, 2)))
# model.add(Dropout(0.5))

# model.add(Conv3D(filters = 512, kernel_size = (3, 3, 3), padding = 'same'))
# model.add(BatchNormalization())
# model.add(AveragePooling3D((1, 2, 2)))
# model.add(Dropout(0.5)) 

# model.add(ConvLSTM2D(filters = 128, kernel_size = (3, 3), padding = 'same', return_sequences = False))
# model.add(AveragePooling2D((2, 2)))
# model.add(Dropout(0.5))
model.add(ConvLSTM2D(filters = 16, kernel_size = (3, 3), padding='same',return_sequences = True, data_format = "channels_last", 
                     input_shape = (seq_len, img_height, img_width,3)))
model.add(ConvLSTM2D(filters =16, kernel_size = (3, 3), return_sequences =False,))
model.add(BatchNormalization())
model.add(Dropout(0.2))
# model.add(ConvLSTM2D(filters =16, kernel_size = (3, 3), return_sequences =False,))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(ConvLSTM2D(filters =16, kernel_size = (3, 3), return_sequences = False,))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(ConvLSTM2D(filters =16, kernel_size = (3, 3), return_sequences = False,))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(len(classes), activation = "softmax"))
 
model.summary()
opt = tfa.optimizers.LazyAdam(lr=0.0001)
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2"], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

# with mirrored_strategy.scope():
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)
stop = EarlyStopping(monitor='val_loss', patience =5,
                      verbose=0, mode='auto', baseline=None, 
                      restore_best_weights=False)
reduce=lr = ReduceLROnPlateau( monitor='val_loss',
                        factor=0.5,
                        patience=2,
                        verbose=0,
                        mode='auto',
                        min_delta=0.0001,
                        cooldown=0,
                        min_lr=0.00000001,)
callbacks = [checkpoint,stop,reduce]
# bs = 8
history_list=[]
loss_history_list=[]
history = model.fit(x = X_train, y = y_train, class_weight=None,epochs=400, batch_size = BS , shuffle=True, validation_data=(X_test,y_test),verbose=1,callbacks=callbacks)

# history = model.fit_generator(
#     train,
#     # steps_per_epoch=75,
#     validation_data=val,
#     # validation_steps=10,
#     verbose=1,
#     epochs=400,
#     callbacks=callbacks
# )

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='best')
plt.savefig('results/accuracy.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='best')
plt.savefig('results/our-loss.png')
plt.show()
