import torch
import torch.nn as nn
import network
import torchvision.transforms as tvt
import matplotlib.pyplot as plt
import dataloader
import copy
import sklearn.metrics
import os
import sys
import glob
import seaborn
import cv2

yolo_interval = 20


def test(net, transform, device,
         batch_size, data_path, save_path,
         jsonpath, resized_data_path, model_path):
    coco_data = dataloader.hw6_dataloader(datapath=data_path, cocodatapath=jsonpath,
                                          transform=transform, resize_w=128, resize_h=128, resized_path=resized_data_path)

    test_data = torch.utils.data.DataLoader(coco_data,
                     batch_size=batch_size, shuffle=False, num_workers=2)
    print("test data len: ",len(test_data))

    net = copy.deepcopy(net)
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    net.eval()

    print("\n\nStarting testing...")
    output_total = []
    label_total = []
    exist = 0
    classify = 0
    misclassify = 0
    notexist = 0
    num_anchor_boxes = 5
    vector_len = 8
    classes = ['cat', 'dog', 'horse', 'none']
    labels_total = []
    classify_total = []

    for i, data in enumerate(test_data):
        img, bbox, category_label, num_objects_in_img, img_path = data
        img = img.to(device).float()
        bbox = bbox.to(device)
        # print(bbox)
        category_label = category_label.to(device)

        num_cells_w = img.shape[-1] // yolo_interval
        num_cells_h = img.shape[-2] // yolo_interval
        num_cells = num_cells_w * num_cells_h

        # initialize GT
        GT = torch.zeros(batch_size, num_cells, num_anchor_boxes, vector_len + 1).to(device)  # B*36*5*9
        for batch in range(batch_size):
            for cell in range(num_cells):
                for anchor in range(num_anchor_boxes):
                    GT[batch, cell, anchor, 8] = 1

        for img_idx in range(img.shape[0]):
            # print(num_objects_in_img)
            # print(img_idx)
            # print(num_objects_in_img[img_idx])
            for obj_idx in range(int(num_objects_in_img[img_idx])):
                height_center_bb = bbox[img_idx][obj_idx][1].item() + \
                                   (bbox[img_idx][obj_idx][3].item() / 2)  # y + height/2
                width_center_bb = bbox[img_idx][obj_idx][0].item() + \
                                  (bbox[img_idx][obj_idx][2].item() / 2)  # x _ width/2
                obj_bb_height = bbox[img_idx][obj_idx][3].item()
                obj_bb_width = bbox[img_idx][obj_idx][2].item()
                if (obj_bb_height < 4) or (obj_bb_width < 4):
                    continue
                AR = float(obj_bb_height) / float(obj_bb_width)
                cell_row_idx = width_center_bb // yolo_interval
                cell_col_idx = height_center_bb // yolo_interval
                cell_row_idx = 5 if cell_row_idx > 5 else cell_row_idx
                cell_col_idx = 5 if cell_col_idx > 5 else cell_col_idx
                if AR <= 0.2:
                    # anchbox = anchor_boxes_1_5[cell_row_idx][cell_col_idx]
                    anchor_number = 0
                elif AR <= 0.5:
                    # anchbox = anchor_boxes_1_3[cell_row_idx][cell_col_idx]
                    anchor_number = 1
                elif AR <= 1.5:
                    # anchbox = anchor_boxes_1_1[cell_row_idx][cell_col_idx]
                    anchor_number = 2
                elif AR <= 4:
                    # anchbox = anchor_boxes_3_1[cell_row_idx][cell_col_idx]
                    anchor_number = 3
                elif AR > 4:
                    # anchbox = anchor_boxes_5_1[cell_row_idx][cell_col_idx]
                    anchor_number = 4

                bh = float(obj_bb_height) / float(yolo_interval)
                bw = float(obj_bb_width) / float(yolo_interval)
                # obj_center_x = float(bbox[img_idx][obj_idx][2].item() +
                #                      (bbox[img_idx][obj_idx][0].item() / 2.0))
                # obj_center_y = float(bbox[img_idx][obj_idx][3].item() +
                #                      (bbox[img_idx][obj_idx][1].item() / 2.0))
                yolocell_center_i = cell_row_idx * yolo_interval + float(yolo_interval) / 2.0
                yolocell_center_j = cell_col_idx * yolo_interval + float(yolo_interval) / 2.0
                del_x = float(width_center_bb - yolocell_center_i) / yolo_interval
                del_y = float(height_center_bb - yolocell_center_j) / yolo_interval
                yolo_vector = [1, del_x, del_y, bw, bh, 0, 0, 0, 0]
                yolo_vector[5 + int(category_label[img_idx][obj_idx].item())] = 1
                yolo_cell_index = cell_row_idx * num_cells_w + cell_col_idx
                # print(yolo_vector)
                GT[img_idx, int(yolo_cell_index), anchor_number] = torch.FloatTensor(yolo_vector)

        GT_flattened = GT.view(img.shape[0], -1)  # B*(1440+5*36)
        outputs = net(img)  # B*(1440+5*36)
        predictions = outputs.view(batch_size, num_cells, num_anchor_boxes, 9)
        GT = GT_flattened.view(batch_size, num_cells, num_anchor_boxes, 9)

        for b in range(batch_size):
            img = cv2.imread(img_path[0])
            for n in range(num_cells):
                for nu in range(num_anchor_boxes):
                    # print(nn.Sigmoid()(predictions[b,n,nu,0]))
                    # if nn.Sigmoid()(predictions[b,n,nu,0])>=0.5:
                    #     print("predictions", nn.Sigmoid()(predictions[b,n,nu,0]))
                    #     print("GT", GT[b,n,nu,0])
                    if (nn.Sigmoid()(predictions[b,n,nu,0])>=0.01 and GT[b,n,nu,0]==1) or (nn.Sigmoid()(predictions[b,n,nu,0])<0.01 and GT[b,n,nu,0]!=1):
                        exist += 1
                    else: notexist += 1
                    if nn.Sigmoid()(predictions[b,n,nu,0])>=0.01 and GT[b,n,nu,0]==1:
                        outputs2 = predictions[b,n,nu,1:5]
                        outputs3 = predictions[b,n,nu,5:9]
                        text_output = classes[torch.argmax(outputs3)]
                        GT2 = GT[b,n,nu,1:5]
                        GT3 = GT[b,n,nu,5:9]
                        text_GT = classes[torch.argmax(GT3)]
                        row_idx = n//6
                        col_idx = n%6

                        yolocell_center_row = row_idx * yolo_interval + float(yolo_interval) / 2.0
                        yolocell_center_col = col_idx * yolo_interval + float(yolo_interval) / 2.0

                        pred_obj_center_row = outputs2[0] * yolo_interval + yolocell_center_row
                        pred_obj_center_col = outputs2[1] * yolo_interval + yolocell_center_col
                        pred_bh = outputs2[2] * yolo_interval
                        pred_bw = outputs2[3] * yolo_interval
                        output_draw = [pred_obj_center_row, pred_obj_center_col, pred_bw, pred_bh]

                        obj_center_row = GT2[0] * yolo_interval + yolocell_center_row
                        obj_center_col = GT2[1] * yolo_interval + yolocell_center_col
                        real_bh = GT2[2] * yolo_interval
                        real_bw = GT2[3] * yolo_interval
                        GT_draw = [obj_center_row, obj_center_col, real_bw, real_bh]

                        cv2.rectangle(img, (int(output_draw[0] - output_draw[2] / 2), int(output_draw[1] - output_draw[3] / 2)),
                                      (int(output_draw[0] + output_draw[2] / 2), int(output_draw[1] + output_draw[3] / 2)),
                                      (0, 0, 255), 1)
                        cv2.putText(img, text_output,
                                    (int(output_draw[0] - output_draw[2] / 2),
                                         int(output_draw[1] - output_draw[3] / 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                                    color=(0, 0, 255), thickness=1)

                        cv2.rectangle(img, (int(GT_draw[0] - GT_draw[2] / 2), int(GT_draw[1] - GT_draw[3] / 2)),
                                      (int(GT_draw[0] + GT_draw[2] / 2),
                                       int(GT_draw[1] + GT_draw[3] / 2)),
                                      (0, 255, 0), 1)
                        cv2.putText(img, text_GT,
                                    (int(GT_draw[0] - GT_draw[2] / 2),
                                     int(GT_draw[1] - GT_draw[3] / 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                                    color=(0, 255, 0), thickness=1)

                        # cv2.rectangle(img, (int(bbox[0][nu][0]), int(bbox[0][nu][1])),
                        #               (int(bbox[0][nu][0]+bbox[0][nu][2]), int(bbox[0][nu][1]+bbox[0][nu][3])), (0, 255, 0),
                        #               1)
                        # cv2.putText(img, text_GT,
                        #             (int(bbox[0][nu][0]), int(bbox[0][nu][1])),
                        #             cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        #             color=(0, 255, 0), thickness=1)


                        # if torch.argmax(nn.Softmax()(outputs3))<3:
                        #     print(nn.Softmax()(outputs3))
                        # if torch.argmax(GT3)<3:
                        #     print(GT3)
                        # if torch.argmax(nn.Softmax()(outputs3)) == torch.argmax(GT3):
                        #     classify += 1
                        # else:
                        #     misclassify += 1
                        labels_total.append(torch.argmax(GT3).cpu())
                        classify_total.append(torch.argmax(nn.Softmax()(outputs3)).cpu())

            if not os.path.exists(save_path + 'tested_imgs/'):
                os.makedirs(save_path + 'tested_imgs/')
            cv2.imwrite(save_path + 'tested_imgs/' + img_path[0][-16:], img)
            # print("save an img to:" + save_path +
            #       'tested_imgs/' + img_path[0][-16:])

    acc_exist = exist / (exist+notexist) * 100
    print("acc_exist = "+ str(acc_exist) + "%")
    # acc_classify = classify / (classify+misclassify) * 100
    # print("acc_classify = " + str(acc_classify) + "%")

    # calculate confusion matrix with sklearn module
    confus_matrix = sklearn.metrics.confusion_matrix(
        labels_total, classify_total, labels=[0, 1, 2, 3])
    print(confus_matrix)

    # classification accuracy calculation
    acc = 0
    for i in range(confus_matrix.shape[0]):
        acc += confus_matrix[i][i]
    Accuracy = acc / confus_matrix.sum() * 100
    print('Accuracy:'+str(Accuracy)+'%')

    # plot confusion matrix
    plt_labels = classes
    plt.figure(figsize = (10,7))
    seaborn.heatmap(confus_matrix, annot=True, fmt= 'd', linewidths = .5,
                    xticklabels= plt_labels, yticklabels= plt_labels)
    plt.title("Net" + 'Accuracy:'+str(Accuracy)+'%')
    plt.savefig(save_path + "net_confusion_matrix.jpg")



if __name__ == '__main__':

    # settings
    device  = torch.device('cuda:1')
    transform = tvt.Compose([tvt.ToTensor(),
                             tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_path = "/home/yangbj/695/PEC/hw5/COCO_Download/Val"
    save_path = "/home/yangbj/695/hw6/"
    jsonpath = "/home/yangbj/695/PEC/hw4/annotations/instances_val2014.json"
    resized_data_path = "/home/yangbj/695/hw6/resized_Val/"
    batch_size = 1
    resize = [128, 128]
    net = network.MODL()
    model_path = '/home/yangbj/695/hw6/MODL4.pth'

    # Run test
    test(net= net,
         transform = transform, device = device, batch_size = batch_size,
         data_path = dataset_path, save_path = save_path,
         jsonpath = jsonpath,
         resized_data_path = resized_data_path, model_path = model_path)
