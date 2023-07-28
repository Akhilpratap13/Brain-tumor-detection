import  cv2
from PIL import Image
from keras.models import load_model
import numpy as np
model = load_model("BrainTumorTrainedModel.h5")
image=cv2.imread("no/No21.jpg")
img=Image.fromarray(image)
img=img.resize((64,64))
img=np.array(img)
input_img=np.expand_dims(img,axis=0)
result=model.predict(input_img)
result=result[0][0]
if result>0:
    print("The person is infected")
else:
    print("The person is not infected")
