import torch
import torch.nn as nn
import network
import torchvision.transforms as tvt
import matplotlib.pyplot as plt
import dataloader
import copy

# class Anchorbox():
#     def __init__(self, AR, tlc, ab_height, ab_width):
#         self.AR = AR
#         self.tlc = tlc
#         self.ab_height = ab_height
#         self.ab_width = ab_width

# from torchvision.ops.boxes import _box_inter_union
# def giou_loss(input_boxes, target_boxes, eps=1e-7):
#     """
#     Args:
#         input_boxes: Tensor of shape (N, 4) or (4,).
#         target_boxes: Tensor of shape (N, 4) or (4,).
#         eps (float): small number to prevent division by zero
#     """
#     inter, union = _box_inter_union(input_boxes, target_boxes)
#     iou = inter / union
#
#     # area of the smallest enclosing box
#     min_box = torch.min(input_boxes, target_boxes)
#     max_box = torch.max(input_boxes, target_boxes)
#     area_c = (max_box[:, 2] - min_box[:, 0]) * (max_box[:, 3] - min_box[:, 1])
#
#     giou = iou - ((area_c - union) / (area_c + eps))
#
#     loss = 1 - giou
#
#     return loss.sum()

def train(transform, device, resized_path, lr = 1e-3, momentum = 0.9, epochs = 10,
          batch_size = 10, data_path = "/home/yangbj/695/YANG/hw5/COCO_Download/Train",
          save_path = "/home/yangbj/695/hw6/",
          cocodatapath = "/home/yangbj/695/YANG/hw4/annotations/instances_train2017.json",
          num_anchor_boxes = 5, vector_len = 8, yolo_interval = 20):

    coco_data = dataloader.hw6_dataloader(datapath=data_path, cocodatapath=cocodatapath,
                                     transform=transform, resize_w=128, resize_h=128, resized_path = resized_path)
    train_data = torch.utils.data.DataLoader(coco_data, batch_size=batch_size,
                                             shuffle=True, num_workers=2)
    net = network.MODL()
    net = copy.deepcopy(net)
    # net.load_state_dict(torch.load("/home/yangbj/695/hw5/net50.pth"))
    net = net.to(device)
    running_loss1 = []
    running_loss2 = []
    running_loss3 = []
    loss_item1 = 0
    loss_item2 = 0
    loss_item3 = 0
    criterion1 = nn.BCELoss()
    criterion2 = nn.MSELoss()
    criterion3 = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999),
                                 eps=1e-08, weight_decay=0, amsgrad=False)

    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    print("\n\nStarting training...")
    for epoch in range(epochs):
        print("")
        for i, data in enumerate(train_data):
            img, bbox, category_label, num_objects_in_img, _ = data
            img = img.to(device).float()
            bbox = bbox.to(device)
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
            # bbox = torch.transpose(torch.stack(bbox), 0, 1)
            # bbox = bbox.to(torch.float32).to(device)
            # Construct the anchor boxes
            # anchor_boxes_1_1 = [[AnchorBox("1/1", (i * yolo_interval, j * yolo_interval),yolo_interval,yolo_interval)
            #                      for i in range(0, num_cells_image_height)] for j in range(0, num_cells_image_width)]
            # anchor_boxes_1_3 = [[AnchorBox("1/3", (i * yolo_interval, j * yolo_interval), yolo_interval, 3*yolo_interval)
            #                      for i in range(0, num_cells_image_height)] for j in range(0, num_cells_image_width)]
            # anchor_boxes_1_5 = [[AnchorBox("1/5", (i * yolo_interval, j * yolo_interval), yolo_interval, 5*yolo_interval)
            #                      for i in range(0, num_cells_image_height)] for j in range(0, num_cells_image_width)]
            # anchor_boxes_5_1 = [[AnchorBox("5/1", (i * yolo_interval, j * yolo_interval), 5*yolo_interval, yolo_interval)
            #                      for i in range(0, num_cells_image_height)] for j in range(0, num_cells_image_width)]
            # anchor_boxes_3_1 = [[AnchorBox("3/1", (i * yolo_interval, j * yolo_interval), 3*yolo_interval, yolo_interval)
            #                      for i in range(0, num_cells_image_height)] for j in range(0, num_cells_image_width)]

            for img_idx in range(img.shape[0]):
                # print(num_objects_in_img)
                # print(img_idx)
                for obj_idx in range(int(num_objects_in_img[img_idx])):
                    # print(bbox[img_idx][obj_idx])
                    # print(category_label[img_idx][obj_idx])
                    height_center_bb = bbox[img_idx][obj_idx][1].item() + \
                                       (bbox[img_idx][obj_idx][3].item() / 2)  # y + height/2
                    width_center_bb = bbox[img_idx][obj_idx][0].item() +\
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

            GT_flattened = GT.view(img.shape[0], -1)  #B*(1440+5*36)
            optimizer.zero_grad()
            outputs = net(img) #B*(1440+5*36)
            # BCE Loss
            loss1 = 0.0
            loss2 = 0.0
            loss3 = 0.0
            for ind in range(0, outputs.shape[1], 9):
                outputs1 = nn.Sigmoid()(outputs[:,ind])
                GT1 = GT_flattened[:,ind]
                loss1 += criterion1(outputs1, GT1)
                # loss_item1 += loss1
                outputs2 = outputs[:,ind+1:ind+5]
                GT2 = GT_flattened[:,ind+1:ind+5]
                loss2 += criterion2(outputs2, GT2)
                # loss_item2 += loss2
                outputs3 = outputs[:, ind + 5:ind+9]
                GT3 = GT_flattened[:, ind + 5:ind+9]
                # print(img.shape[0])
                for b in range(img.shape[0]):
                    if b==0:
                        class_label = torch.unsqueeze(torch.argmax(GT3[b,:]), 0)
                    else:
                        class_label = torch.cat((class_label, torch.unsqueeze(torch.argmax(GT3[b,:]), 0)))
                loss3 += criterion3(outputs3, class_label)

            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()
            loss_item1 += loss1.item()
            loss_item2 += loss2.item()
            loss_item3 += loss3.item()
            per = 8
            if (i+1) % per == 0:
                print("[epoch:%d, batch:%5d, lr: %.9f] loss1: %.3f" %
                      (epoch + 1, i + 1, lr,  loss_item1 / float(per)))
                print("[epoch:%d, batch:%5d, lr: %.9f] loss2: %.3f" %
                      (epoch + 1, i + 1, lr, loss_item2 / float(per)))
                print("[epoch:%d, batch:%5d, lr: %.9f] loss3: %.3f" %
                      (epoch + 1, i + 1, lr, loss_item3 / float(per)))
                running_loss1.append(loss_item1/float(per))
                running_loss2.append(loss_item2 / float(per))
                running_loss3.append(loss_item3/float(per))
                loss_item1, loss_item2, loss_item3 = 0.0, 0.0, 0.0
    torch.save(net.state_dict(), save_path+'MODL4.pth')

    return running_loss1, running_loss2, running_loss3


if __name__ == '__main__':
    device  = torch.device('cuda:1')
    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset_path = "/home/yangbj/695/PEC/hw5/COCO_Download/Train"
    coco_json = "/home/yangbj/695/PEC/hw4/annotations/instances_train2017.json"
    save_path = "/home/yangbj/695/hw6/"
    resized_path = "/home/yangbj/695/hw6/resized/"
    running_loss1, running_loss2, running_loss3 = train(transform = transform,
    device = device, resized_path=resized_path,
    lr = 1e-4, momentum = 0.9, epochs = 50, batch_size = 8,
    data_path = trainset_path, save_path = save_path,
    cocodatapath = "/home/yangbj/695/PEC/hw4/annotations/instances_train2017.json")

    plt.figure()
    plt.title('BCE Loss')
    plt.xlabel('Per 8 Iterations')
    plt.ylabel('Loss')
    plt.plot(running_loss1, label = 'BCE Loss')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig(save_path + "train3_loss1" + ".jpg")


    plt.figure()
    plt.title('MSE Loss')
    plt.xlabel('Per 8 Iterations')
    plt.ylabel('Loss')
    plt.plot(running_loss2, label='MSE Loss')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig(save_path + "train3_loss2" + ".jpg")

    plt.figure()
    plt.title('CE Loss')
    plt.xlabel('Per 8 Iterations')
    plt.ylabel('Loss')
    plt.plot(running_loss3, label='CE Loss')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig(save_path + "train3_loss3" + ".jpg")
