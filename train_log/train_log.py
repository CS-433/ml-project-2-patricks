# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 19:45:22 2021

@author: syson
"""
'''
[2021-12-12 11:27:46,851][line: 81] ==> train Epoch:[285/300]  loss=0.00021  acc_hard=1.000 acc_soft=1.000 lr=0.0005545
[2021-12-12 11:27:50,816][line: 81] ==> val Epoch:[285/300]  loss=0.92450  acc_hard=0.692 acc_soft=0.924 lr=0.0005545
'''
#0->RandomHorizontalFlip
#1->RandomVerticalFlip
#2->RandomRotation
#3->RandomResizedCrop
#4->RandomPerspective
#5->GaussianBlur
#6->RandomAdjustSharpness
#7->RandomChoice
#8->Normailzie
#9->Original Transform
import matplotlib.pyplot as plt
'''
tags = ["RandomHorizontalFlip","RandomVerticalFlip",
        "RandomRotation","RandomResizedCrop",
        "RandomPerspective","GaussianBlur",
        "RandomAdjustSharpness","RandomChoice",
        "Normailzie","Original Transform"]

#tags_150_epoch:2,3,4,5
tags_150_epoch = ["RandomRotation_150_epoch","RandomResizedCrop_150_epoch",
                  "RandomPerspective_150_epoch","GaussianBlur_150_epoch"]
'''
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
#plt.rcParams['figure.figsize'] = (12.0, 16.0)#unit:6,4
plt.rcParams['figure.figsize'] = (6.0, 4.0)
plt.figure()
#plt.rcParams['figure.figsize'] = (6.0, 4.0)
plt.subplots_adjust(wspace = 0.5, hspace =1)#调整子图间距
i = 1
#for tag in tags_150_epoch:
#srcFile = open('ZenNas_alpha_banlanced_loss.log', 'r+')
srcFile = open('ZenNas_alpha_banlanced_loss.log', 'r+')
lines = srcFile.readlines()

train_loss = []
val_loss = []
train_hard_acc = []
val_hard_acc = []
last_train_epoch = 0
last_val_epoch = 0
for line in lines:
    data = line.split()
    loss = (float)(data[6].split('=')[1])
    acc = (float)(data[7].split('=')[1])
    if data[4] == 'train':
        rec_train_epoch = (int)(data[5].split('/')[0].split('[')[1])
        if rec_train_epoch > 200:
            break
        if rec_train_epoch != last_train_epoch:
            last_train_epoch = rec_train_epoch
            train_loss.append(loss)
            train_hard_acc.append(acc)
    if data[4] == 'val':
        rec_val_epoch = (int)(data[5].split('/')[0].split('[')[1])
        if rec_val_epoch > 200:
            break
        if rec_val_epoch != last_val_epoch:
            last_val_epoch = rec_val_epoch
            val_loss.append(loss)
            val_hard_acc.append(acc) 

epochs = range(len(train_loss))
#plt.subplot(2,2,i)
plt.plot(epochs, train_hard_acc, 'b', label='train_hard_acc')
plt.plot(epochs, val_hard_acc, 'r', label='val_hard_acc')
#plt.title(tag+'-acc')
plt.title('ZenNas_alpha_banlanced_loss-acc')
plt.legend(loc='lower right',prop = {'size':8})
plt.savefig('./ZenNas_alpha_banlanced_loss-acc.jpg')
#i = i + 1
#plt.subplot(2,2,i)


plt.figure()
plt.plot(epochs, train_loss, 'r', label='train_loss')
plt.plot(epochs, val_loss, 'b', label='val_loss')
#plt.title(tag+'-loss')
plt.title('ZenNas_alpha_banlanced_loss-loss')
plt.legend(prop = {'size':8})
#i = i + 1
plt.savefig('./ZenNas_alpha_banlanced_loss-loss.jpg')
srcFile.close()

