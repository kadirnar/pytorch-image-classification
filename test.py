# Kütüphanelerin Yüklenmesi

import torchvision.transforms as transforms
from PIL import Image
import cv2
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import imutils
import cv2
#%%

labels = ['inmevar','inmeyok']
dataa = ["./test/test_inmeyok_1.jpg","./test/test_inmevar_1.jpg"]
data = dataa[0]

test_transforms = transforms.Compose([transforms.Resize(512),
                                      transforms.CenterCrop(512),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
])

# Model Yükleme

model_ismi ="model.pth"
model = torch.load(model_ismi)
model.eval()
  
# resmi çağırdık
img = Image.open(data)

# resmi işleme 
img = test_transforms(img)
torch.unsqueeze(img,dim=1)

image = cv2.imread(data)
orig = image.copy()

# ikili sınıflandırma yaptığımız için aktivasyon fonksiyonunu softmax olarak aldık

prediction = F.softmax(model(img[None]),dim=1)
prediction = prediction.argmax()
print(labels[prediction]) 
#%%
if prediction >= 0.5:

	label = "InmeYOK"
	a = 100 * prediction

else:

	label = "InmeVAR"
	a = (1-prediction) * 100


label = "{}: {:.2f}%".format(label, a)
# draw the label on the image
output = imutils.resize(orig, width=250)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (255, 0, 0), 2)
# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)