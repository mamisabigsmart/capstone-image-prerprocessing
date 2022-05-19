import torch
import os
import cv2
from yolo.utils.utils import *
from predictors.YOLOv3 import YOLOv3Predictor
#from predictors.DetectronModels import Predictor
import glob
from tqdm import tqdm
import sys



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


#YOLO PARAMS
yolo_df2_params = {   "model_def" : "yolo/df2cfg/yolov3-df2.cfg",
"weights_path" : "yolo/weights/yolov3-df2_15000.weights",
"class_path":"yolo/df2cfg/df2.names",
"conf_thres" : 0.5,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device}

yolo_modanet_params = {   "model_def" : "yolo/modanetcfg/yolov3-modanet.cfg",
"weights_path" : "yolo/weights/yolov3-modanet_last.weights",
"class_path":"yolo/modanetcfg/modanet.names",
"conf_thres" : 0.5,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device}


#DATASET
dataset = 'modanet'


if dataset == 'df2': #deepfashion2
    yolo_params = yolo_df2_params

if dataset == 'modanet':
    yolo_params = yolo_modanet_params


#Classes
classes = load_classes(yolo_params["class_path"])
#print(classes)
#Colors
cmap = plt.get_cmap("rainbow")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 13)])
#np.random.shuffle(colors)



#


model = 'yolo'

if model == 'yolo':
    detectron = YOLOv3Predictor(params=yolo_params)
else:
    detectron = Predictor(model=model,dataset= dataset, CATEGORIES = classes)

#Faster RCNN / RetinaNet / Mask RCNN



while(True):
    path = input('img path: ')
    if not os.path.exists(path):
        print('Img does not exists..')
        continue
    img = cv2.imread(path)
    img_shape = img.shape
    print(img_shape)
    img_center_x = int(img.shape[1]/2)
    img_center_y = int(img.shape[0]/2)

    detections = detectron.get_detections(img)
    #print(detections)
    #detections = yolo.get_detections(img)

    #unique_labels = np.array(list(set([det[-1] for det in detections])))

    #n_cls_preds = len(unique_labels)
    #bbox_colors = colors[:n_cls_preds]
    box_info_storage = []

    
    if len(detections) != 0 :
        # read from bottom to top
        detections.sort(reverse=False ,key = lambda x:x[4])
        for x1, y1, x2, y2, cls_conf, cls_pred in detections:
                
                #feat_vec =detectron.compute_features_from_bbox(img,[(x1, y1, x2, y2)])
                #feat_vec = detectron.extract_encoding_features(img)
                #print(feat_vec)
                #print(a.get_field('features')[0].shape)
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf))           

                
                #color = bbox_colors[np.where(unique_labels == cls_pred)[0]][0]
                color = colors[int(cls_pred)]
                
                color = tuple(c*255 for c in color)
                # color of bounding box
                color = (.7*color[2],.7*color[1],.7*color[0])
                # font of bounding box
                font = cv2.FONT_HERSHEY_SIMPLEX   
            
            
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                text =  "%s conf: %.3f" % (classes[int(cls_pred)] ,cls_conf)
                # x1,y1 left upper corner, x2,y2 right lower corner
                print(x1,y1,x2,y2)

                # get the center of bounding box
                center_rect_x = int((x2-x1)/2+x1)
                center_rect_y = int((y2-y1)/2+y1)
                print("center is:",center_rect_x,center_rect_y)

                # calculate euclidean distance of bounding box center and img center
                center_distance = (center_rect_x - img_center_x)**2 + (center_rect_y - img_center_y)**2
                print(center_distance)

                box_info = [[x1,y1,x2,y2],[center_rect_x,center_rect_y],center_distance,classes[int(cls_pred)]]

                box_info_storage.append(box_info)

                cv2.rectangle(img,(x1,y1) , (x2,y2) , color,3)
                y1 = 0 if y1<0 else y1
                y1_rect = y1-25
                y1_text = y1-5

                if y1_rect<0:
                    y1_rect = y1+27
                    y1_text = y1+20
                cv2.rectangle(img,(x1-2,y1_rect) , (x1 + int(8.5*len(text)),y1) , color,-1)
                cv2.putText(img,text,(x1,y1_text), font, 0.5,(255,255,255),1,cv2.LINE_AA)

    min_distance_box_index = 0
    min_distance = 10000000
    print(box_info_storage)
    for box in box_info_storage:
        if box[2] < min_distance:
            min_distance_box_index = box_info_storage.index(box)
    print("The label of this picture is: ", box_info_storage[min_distance_box_index][3])
                
    cv2.imshow('Detections',img)
    img_id = path.split('/')[-1].split('.')[0]
    cv2.imwrite('output/ouput-test_{}_{}_{}.jpg'.format(img_id,model,dataset),img)
    cv2.waitKey(0)