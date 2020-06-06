import glob
import json
import os


test = {'label_names': [] , 'image_names':[] , 'image_labels':[]}
pathname = os.getcwd()
#pathname = pathname.split('filelists')[0]
print(pathname)

f = open(pathname + '/cifar-FS/splits/bertinetto/test.txt')
classes = f.readlines()

count = 80
for each in classes: 
	each = each.strip()
	test['label_names'].append(each)
	files = glob.glob( pathname + '/cifar-FS/data/' + each + '/*')
	for image_name in files:
		test['image_names'].append( image_name)
		test['image_labels'].append(count)
	count +=1


json.dump(test , open('novel.json','w')) 



base = {'label_names': [] , 'image_names':[] , 'image_labels':[]}
f = open(pathname + '/cifar-FS/splits/bertinetto/train.txt')
classes = f.readlines()

count = 0
for each in classes: 
	each = each.strip()
	base['label_names'].append(each)
	files = glob.glob( pathname + '/cifar-FS/data/' + each + '/*')
	for image_name in files:
		base['image_names'].append( image_name)
		base['image_labels'].append(count)
	count +=1


json.dump(base , open('base.json','w')) 



val = {'label_names': [] , 'image_names':[] , 'image_labels':[]}
f = open(pathname + '/cifar-FS/splits/bertinetto/val.txt')
classes = f.readlines()

count = 0
for each in classes: 
	each = each.strip()
	val['label_names'].append(each)
	files = glob.glob( pathname + '/cifar-FS/data/' + each + '/*')
	for image_name in files:
		val['image_names'].append( image_name)
		val['image_labels'].append(count)
	count +=1


json.dump(val , open('val.json','w')) 
