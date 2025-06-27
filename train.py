import cv2
import numpy as np
from PIL import Image
data = []
label = []
num_people = 2  # Số người, chỉnh theo số người thực tế
num_images = 20 # Số ảnh mỗi người
for i in range(1, num_people+1):
    for j in range(1, num_images+1):
        filename = f'd:/TriTueNhanTao/NhanDienKhuonMat/datasetIM/anh{i}.{j}.jpg'
        Img = cv2.imread(filename)
        if Img is None:
            print(f"Không đọc được {filename}")
            continue
        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
        Img = cv2.resize(src=Img, dsize=(100,100))
        Img = np.array(Img)
        data.append(Img)
        label.append(i-1)  # Mỗi người 1 label khác nhau
data1 = np.array(data)
label = np.array(label)
data1 = data1.reshape((-1,100,100,1))
X_train = data1/255
from tensorflow.keras.utils import to_categorical
trainY = to_categorical(label, num_people)
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
Model = Sequential()
shape = (100,100, 1)
Model.add(Conv2D(32,(3,3),padding="same",input_shape=shape))
Model.add(Activation("relu"))
Model.add(Conv2D(32,(3,3), padding="same"))
Model.add(Activation("relu"))
Model.add(MaxPooling2D(pool_size=(2,2)))
Model.add(Conv2D(64,(3,3), padding="same"))
Model.add(Activation("relu"))
Model.add(MaxPooling2D(pool_size=(2,2)))
Model.add(Flatten())
Model.add(Dense(512))
Model.add(Activation("relu"))
Model.add(Dense(num_people))
Model.add(Activation("softmax"))
Model.summary()
Model.compile(loss='categorical_crossentropy',
              optimizer=SGD(learning_rate=0.01),
              metrics=['accuracy'])
print("start training")
Model.fit(X_train, trainY, epochs=100, batch_size=8, verbose=1)
Model.save('khuonmat.h5')
print('Đã lưu model vào khuonmat.h5')
