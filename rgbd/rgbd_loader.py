#coding:utf-8

def load_data(dataset):
   # class1_files = []
   # class_labels = []
   # with open('rgbd/color.txt') as f:
   #     for line in f.readlines():
   #         data = line.strip().split(" ")
   #         class1_files.append(data[0])
   #         if data[1]=='999':
   #             class_labels.append(546)
   #         else:
   #             class_labels.append(int(data[1]))
   #
   # class2_files = []
   # with open('rgbd/depth.txt') as f:
   #     for line in f.readlines():
   #         data = line.strip().split(" ")
   #         class2_files.append(data[0])
   #
   # return class1_files, class_labels, class2_files
   class_files = []
   with open('/media/liuhan/xiangziBRL/Lock3DFace/croppedData_LightenedCNN/PairList/test/test_colorLabel.txt') as f:
       for line in f.readlines():
           data = line.strip().split(" ")
           class_files.append(data[0])
   return class_files
