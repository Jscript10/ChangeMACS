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
from OurMethod.ChangeMACS import UNet
# cdd  levir sysu
# parser.add_argument('--hr1_test1', default='CDdata/LEVIR/test/time1', type=str, help='hr image at t1 in validation set')
# parser.add_argument('--hr2_test1', default='CDdata/LEVIR/test/time2', type=str, help='hr image at t2 in validation set')
# parser.add_argument('--lab_test1', default='CDdata/LEVIR/test/label', type=str, help='label image in vn validation set')

parser.add_argument('--hr1_test1', default='CDdata/sysu/test/time1', type=str, help='hr image at t1 in validation set')
parser.add_argument('--hr2_test1', default='CDdata/sysu/test/time2', type=str, help='hr image at t2 in validation set')
parser.add_argument('--lab_test1', default='CDdata/sysu/test/label', type=str, help='label image in vn validation set')


if __name__ == "__main__":

    if not os.path.exists('output_img'):
        os.mkdir('output_img')

    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet().to(device, dtype=torch.float)

    path = 'epoch/sysu/our/xxx.pth'  # the path of the model2
    net.load_state_dict(torch.load(path, map_location=device))

    net.eval()


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
    output_dir = 'testresult'   #预测输出路径
    os.makedirs(output_dir, exist_ok=True)

    for img, file_name in save_img:
        save_path = os.path.join(output_dir, file_name)  # 使用原始文件名
        img[img >= thold] = 255
        img[img < thold] = 0
        cv2.imwrite(save_path, img)