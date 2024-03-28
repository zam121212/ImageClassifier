import tensorflow 
from keras import layers, models
import numpy as np
import onnx
import tf2onnx.convert
import os
import cv2
from sklearn.model_selection import train_test_split

DIR =r"C:\Users\Ganesh Pavan Munduri\DataSet"
CATEGORIES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
len_categories=len(CATEGORIES)
training_data=[]

for categories in CATEGORIES:
    path=os.path.join(DIR,categories)
    class_num=CATEGORIES.index(categories)
    for img in os.listdir(path):
        try:
            img_array=cv2.imread(os.path.join(path,img))
            new_array=cv2.resize(img_array,(32,32))
            training_data.append([new_array,class_num])
        except Exception as e:
            pass
X=[]
y=[]
for features,labels in training_data:
    X.append(features)
    y.append(labels)
X=np.array(X)
y=np.array(y)

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.1,random_state=1)

Xtrain=np.array(Xtrain)
Xtest=np.array(Xtest)

Xtrain = Xtrain / 255.0
Xtest = Xtest / 255.0

print(Xtrain.shape,ytrain)
print(Xtest.shape,ytest)


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len_categories, activation='softmax')
])

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(Xtrain, ytrain, epochs=10)

cnn.evaluate(Xtest,ytest)

y_pred = cnn.predict(Xtest)

print(y_pred)

y_classes = [np.argmax(element) for element in y_pred]

for i in range(len(y_classes)):
    print(CATEGORIES[y_classes[i]])

Pathtosave=r"C:\Users\Ganesh Pavan Munduri\CNN"
cnn.save(Pathtosave)

onnx_model_name = 'model.onnx'

model=models.load_model(Pathtosave)

onnx_model,_ = tf2onnx.convert.from_keras(model)

onnx.save_model(onnx_model,onnx_model_name)