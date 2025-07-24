import os
from tqdm import tqdm
import cv2
import torch.utils.data
from torch.utils.data import DataLoader
from data_utils import LoadDatasetFromFolder_CD
import numpy as np
from configures import parser
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from cdmodel.AchangeCoMa import UNet
# cdd  levir sysu
# parser.add_argument('--hr1_test1', default='CDdata/LEVIR/test/time1', type=str, help='hr image at t1 in validation set')
# parser.add_argument('--hr2_test1', default='CDdata/LEVIR/test/time2', type=str, help='hr image at t2 in validation set')
# parser.add_argument('--lab_test1', default='CDdata/LEVIR/test/label', type=str, help='label image in vn validation set')

parser.add_argument('--hr1_test1', default='CDdata/sysu/test/time1', type=str, help='hr image at t1 in validation set')
parser.add_argument('--hr2_test1', default='CDdata/sysu/test/time2', type=str, help='hr image at t2 in validation set')
parser.add_argument('--lab_test1', default='CDdata/sysu/test/label', type=str, help='label image in vn validation set')

def process_images_all(label_dir, pred_dir, output_dir):
    # 获取label和pred目录下的所有png和jpg文件
    label_files = [f for f in os.listdir(label_dir) if f.endswith(('.png', '.jpg'))]
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg'))]

    # 过滤出两个目录下相同名字的文件
    common_files = set(label_files) & set(pred_files)

    for filename in common_files:
        dir_label = os.path.join(label_dir, filename)
        dir_pred = os.path.join(pred_dir, filename)

        label = Image.open(dir_label)
        pred = Image.open(dir_pred).convert("L")

        label_rgb = pred.convert('RGB')

        label_np = np.array(label)
        pred_np = np.array(pred)

        label_rgb_np = np.array(label_rgb)
        for i in range(label_np.shape[0]):
            for j in range(label_np.shape[1]):
                if label_np[i][j] >= 100 and pred_np[i][j] <= 90:
                    label_rgb_np[i][j] = [255, 0, 0]
                elif label_np[i][j] <= 90 and pred_np[i][j] >= 100:
                    label_rgb_np[i][j] = [0, 255, 0]

        img = Image.fromarray(label_rgb_np)
        save_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)

def process_images(selected_files, label_dir, pred_dir, output_dir):#这个用来生成marked image 颜色
    for filename in selected_files:
        dir_label = os.path.join(label_dir, filename)
        dir_pred = os.path.join(pred_dir, filename)

        label = Image.open(dir_label)
        pred = Image.open(dir_pred).convert("L")

        label_rgb = pred.convert('RGB')

        label_np = np.array(label)
        pred_np = np.array(pred)

        label_rgb_np = np.array(label_rgb)
        for i in range(label_np.shape[0]):
            for j in range(label_np.shape[1]):
                if label_np[i][j] >= 100 and pred_np[i][j] <= 90:
                    label_rgb_np[i][j] = [255, 0, 0]  # Red color
                elif label_np[i][j] <= 90 and pred_np[i][j] >= 100:
                    label_rgb_np[i][j] = [0, 255, 0]  # Green color 0, 255, 0  blue 135, 206, 250

        img = Image.fromarray(label_rgb_np)
        save_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)

if __name__ == "__main__":

    if not os.path.exists('output_img'):
        os.mkdir('output_img')

    args = parser.parse_args()

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device=torch.device('cpu')
    #加载网络
    net = UNet().to(device, dtype=torch.float)
    #加载模型参数
    path = 'epoch/sysu/our/netCD_epoch_198_pre_0.8339_rec_0.8332_f1_0.8335_iou_0.7145_OA_0.9135_KC_0.7640.pth'  # the path of the model2
    net.load_state_dict(torch.load(path, map_location=device))
    # 测试模式
    net.eval()

    #加载图片数据
    test_set = LoadDatasetFromFolder_CD(args, args.hr1_test1, args.hr2_test1, args.lab_test1)
    test_loader = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=1, shuffle=False)
    save_img=[]
    file_names = test_set.file_names
    with torch.no_grad():
        tbar = tqdm(test_loader)
        for idx, (hr_img1, hr_img2, label) in enumerate(tbar):
            hr_img1 = hr_img1.to(device, dtype=torch.float)
            hr_img2 = hr_img2.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float)

            dist = net(hr_img1, hr_img2)
            dist = np.array(dist.data.cpu()[0])[0]
            save_img.append((dist, file_names[idx]))
    #
    thold = 1
    #nthold = 1
    data = '2sysu'
    netname = "_ournet"
    CONSTANT_PART = "_thold_"
    output_dir = f'testresult/{data}/{data}{netname}{CONSTANT_PART}{thold}'   #预测输出路径
    os.makedirs(output_dir, exist_ok=True)

    for img, file_name in save_img:
        save_path = os.path.join(output_dir, file_name)  # 使用原始文件名
        img[img >= thold] = 255
        img[img < thold] = 0
        cv2.imwrite(save_path, img)

    ####运行marked image 指定图像
    #
    # label_dir = f"{data}/test/label"  #标签路径

    label_dir = "CDdata/sysu/test/label"  #标签路径

    pred_dir = f"testresult/{data}/{data}{netname}{CONSTANT_PART}{thold}" #测试结果的路径
    output_dir = f"testresult/1mark/{data}/{data}{netname}{CONSTANT_PART}{thold}" #保存红色绿色路径
    #
    # # 选中的几个图片文件名
    # #selected_files = ['00573.png', '00586.png', '00613.png', '00636.png', '00649.png', '00695.png', '00756.png']
    # # selected_files = ['00006.png', '00019.png', '00114.png', '00146.png', '00283.png', '00982.png', '01014.png',
    # #                   '01117.png', '01494.png'] jpg
    # # selected_files = ['00059.png', '00064.png', '00424.png', '00776.png', '00792.png', '00794.png', '00829.png',
    # #                   '01731.png', '01849.png', '02244.png', '02419.png']
    #
    # selected_files = ['1.png', '2.png', '3.png', '4.png', '5.png', '6.png']
    #
    # process_images(selected_files, label_dir, pred_dir, output_dir)

    ###marked image 所有图像
    process_images_all(label_dir, pred_dir, output_dir)