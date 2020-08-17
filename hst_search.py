import argparse
import time
import os
from sys import platform
import numpy as np
from yolo3.yolo3_model import *
from yolo3.utils.datasets import *
from yolo3.utils.utils import *

def compare_hist(HAST,img):
    img = cv2.resize(img,(256,128))
    hst = cv2.calcHist([img],[1],None,[256],[0,256])
    cv2.normalize(hst,hst,0,255*0.9,cv2.NORM_MINMAX)
    reval = cv2.compareHist(HAST,hst,cv2.HISTCMP_BHATTACHARYYA)
    return reval

def xywh_feature(x,y,w,h):
    feat = np.array([x,y,w,h])
    feat = feat/416.0
    return feat


def detect(cfg,
           data,
           weights,
           images='data/samples',  # input folder
           output='output',  # output folder
           fourcc='mp4v',  # video codec
           img_size=416,
           conf_thres=0.5,
           nms_thres=0.5,
           dist_thres=1.0,
           save_txt=False,
           save_images=True):

    # Initialize
    device = torch_utils.select_device(force_cpu=False)
    torch.backends.cudnn.benchmark = False  # set False for reproducible results
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    ############# 行人重识别模型初始化 #############
    query_feats1 = []
    query_feats2 = []
    query_feats3 = []
    path = "./query"
    for i, img in enumerate(os.listdir(path)):
        if img.split('_')[0]=='0001':
            img = cv2.imread(os.path.join(path,img))
            img =cv2.resize(img,(256,128))
            feat = cv2.calcHist([img],[1],None,[256],[0,256])
            cv2.normalize(feat,feat,0,255*0.9,cv2.NORM_MINMAX)         # 一共2张待查询图片，每张图片特征向量2048 torch.Size([2, 2048])
            query_feats1.append(feat)
        elif img.split('_')[0]=='0002':
            img = cv2.imread(os.path.join(path, img))
            img = cv2.resize(img, (256, 128))
            feat = cv2.calcHist([img], [1], None, [256], [0, 256])
            cv2.normalize(feat, feat, 0, 255 * 0.9, cv2.NORM_MINMAX)
            query_feats2.append(feat)
        else:
            print('please input query img with correct sample')


    print("The query feature is normalized")

    ############# 行人检测模型初始化 #############
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Eval mode
    model.to(device).eval()
    # Half precision
    opt.half = opt.half and device.type != 'cpu'  # half precision only supported on CUDA
    if opt.half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if opt.webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size, half=opt.half)
    else:
        dataloader = LoadImages(images, img_size=img_size, half=opt.half)

    # Get classes and colors
    # parse_data_cfg(data)['names']:得到类别名称文件路径 names=data/coco.names
    classes = load_classes(parse_data_cfg(data)['names']) # 得到类别名列表: ['person', 'bicycle'...]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))] # 对于每种类别随机使用一种颜色画框

    # Run inference
    t0 = time.time()
    for i, (path, img, im0, vid_cap) in enumerate(dataloader):

        t = time.time()
        # if i % 2 == 0:
        save_path = str(Path(output) / Path(path).name) # 保存的路径

        # Get detections shape: (3, 416, 320)
        img = torch.from_numpy(img).unsqueeze(0).to(device) # torch.Size([1, 3, 416, 320])
        pred, _ = model(img) # 经过处理的网络预测，和原始的
        det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0] # torch.Size([5, 7])

        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size 映射到原图
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results to screen image 1/3 data\samples\000493.jpg: 288x416 5 persons, Done. (0.869s)
            print('%gx%g ' % img.shape[2:], end='')  # print image size '288x416'
            for c in det[:, -1].unique():   # 对图片的所有类进行遍历循环
                n = (det[:, -1] == c).sum() # 得到了当前类别的个数，也可以用来统计数目
                if classes[int(c)] == 'person':
                    print('%g %ss' % (n, classes[int(c)]), end=', ') # 打印个数和类别'5 persons'

            # Draw bounding boxes and labels of detections
            # (x1y1x2y2, obj_conf, class_conf, class_pred)
            count = 0
            gallery_img = []
            gallery_loc = []
            idd1 = []
            idd2 = []
            for *xyxy, conf, cls_conf, cls in det: # 对于最后的预测框进行遍历
                # *xyxy: 对于原图来说的左上角右下角坐标: [tensor(349.), tensor(26.), tensor(468.), tensor(341.)]
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf) # 'person 1.00'
                if classes[int(cls)] == 'person':
                    #plot_one_bo x(xyxy, im0, label=label, color=colors[int(cls)])
                    xmin = int(xyxy[0])
                    ymin = int(xyxy[1])
                    xmax = int(xyxy[2])
                    ymax = int(xyxy[3])
                    w = xmax - xmin # 233
                    h = ymax - ymin # 602
                    # 如果检测到的行人太小了，感觉意义也不大
                    # 这里需要根据实际情况稍微设置下
                    if w*h > 500:
                        scores1=[]
                        scores2 = []
                        scores3 = []
                        gallery_loc.append((xmin, ymin, xmax, ymax))
                        crop_img = im0[ymin:ymax, xmin:xmax] # HWC (602, 233, 3)
                        if query_feats1:
                            for HAST1 in query_feats1:
                                score = compare_hist(HAST1,crop_img)
                                scores1.append(score)
                            score = np.mean(scores1)
                            idd1.append(score)   #idd为1X7的数组，代表id_1与7个crop_img的‘距离’。
                        if query_feats2:
                            for HAST2 in query_feats2:
                                score = compare_hist(HAST2,crop_img)
                                scores2.append(score)
                            score = np.mean(scores2)
                            idd2.append(score)   #idd为1X7的数组，代表id_2与7个crop_img的‘距离’。
            index1=1000
            if idd1:
                dist1 = min(idd1)
                if dist1< 0.25:
                    index1 = np.argmin(idd1)
                    print("hua chu 0001 bbox")
                    plot_one_box(gallery_loc[index1], im0, label='find_0001', color=colors[int(cls)])
                    # cv2.imshow('person search', im0)
                    # cv2.waitKey()
            if idd2:
                dist2 = min(idd2)
                if dist2 < 0.25:
                    index2 = np.argmin(idd2)
                    if index2==index1:
                        del_bbox =gallery_loc.pop(index1)
                        del_elm = idd2.pop(index1)
                        dist2 = min(idd2)
                        if dist2<0.25:
                            index2 = np.argmin(idd2)
                            print("find 0002 bbox")
                            plot_one_box(gallery_loc[index2], im0, label='find_0002', color=colors[int(cls)])
                    else:
                        print("hua chu 0002 bbox")
                        plot_one_box(gallery_loc[index2], im0, label='find_0002', color=colors[int(cls)])
                        # cv2.imshow('person search', im0)
                        # cv2.waitKey()
        print('Done. (%.3fs)' % (time.time() - t))

        if opt.webcam:  # Show live webcam
            cv2.imshow(weights, im0)

        if save_images:  # Save image with detections
            if dataloader.mode == 'images':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
                vid_writer.write(im0)

    if save_images:
        print('Results saved to %s' % os.getcwd() + os.sep + output)
        if platform == 'darwin':  # macos
            os.system('open ' + output + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolo3/cfg/yolov3.cfg', help="模型配置文件路径")
    parser.add_argument('--data', type=str, default='yolo3/data/coco.data', help="数据集配置文件所在路径")
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='模型权重文件路径')
    parser.add_argument('--images', type=str, default='input_video', help='需要进行检测的图片文件夹')
    parser.add_argument('-q', '--query', default=r'query', help='查询图片的读取路径.')
    parser.add_argument('--img-size', type=int, default=416, help='输入分辨率大小')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='物体置信度阈值')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='NMS阈值')
    parser.add_argument('--dist_thres', type=float, default=1.0, help='行人图片距离阈值，小于这个距离，就认为是该行人')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='fourcc output video codec (verify ffmpeg support)')
    parser.add_argument('--output', type=str, default='output', help='检测后的图片或视频保存的路径')
    parser.add_argument('--half', default=False, help='是否采用半精度FP16进行推理')
    parser.add_argument('--webcam', default=False, help='是否使用摄像头进行检测')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt.cfg,
               opt.data,
               opt.weights,
               images=opt.images,
               img_size=opt.img_size,
               conf_thres=opt.conf_thres,
               nms_thres=opt.nms_thres,
               dist_thres=opt.dist_thres,
               fourcc=opt.fourcc,
               output=opt.output)