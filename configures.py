import argparse

#training options
parser = argparse.ArgumentParser(description='Train SRCDNet')

# training parameters
parser.add_argument('--num_epochs', default=300, type=int, help='train epoch number')
parser.add_argument('--batchsize', default=2, type=int, help='batchsize')
parser.add_argument('--val_batchsize', default=2, type=int, help='batchsize for validation')
parser.add_argument('--num_workers', default=0, type=int, help='num_workers')
parser.add_argument('--n_class', default=2, type=int, help='number of class')
parser.add_argument('--gpu_id', default="0", type=str, help='which gpu to run.')                                            
parser.add_argument('--suffix', default=['.png','.jpg'], type=list, help='the suffix of the image files.')
parser.add_argument("--image_size", type=int, default=256, help="image_size")  #训练1024  256 输入
parser.add_argument('--img_size', default=256, type=int, help='batchsize for validation')
parser.add_argument('--lr', type=float, default=0.000125, help='initial learning rate for CDNet')
parser.add_argument('--w_cd', type=float, default=0.001, help='factor to balance the weight of CD loss in Generator loss')
parser.add_argument('--scale', default=8, type=int, help='resolution difference between images. [ 2| 4| 8]')
parser.add_argument("--resume", type=str, default=None, help="load resume")
parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")  # vit_b      vit_h
parser.add_argument("--sam_checkpoint", type=str, default="models/sam_vit_b_01ec64.pth",help="sam checkpoint")  # sam_vit_b_01ec64.pth
parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
parser.add_argument("--iter_point", type=int, default=8, help="point iterations")
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
# path for loading data from folder    /home/jiang/SRCD/data

#cdd
# parser.add_argument('--hr1_train', default='CDD/train/time1', type=str, help='hr image at t1 in training set')
# parser.add_argument('--hr2_train', default='CDD/train/time2', type=str, help='hr image at t2 in training set')
# parser.add_argument('--lab_train', default='CDD/train/label', type=str, help='label image in training set')
#
# parser.add_argument('--hr1_val', default='CDD/val/time1', type=str, help='hr image at t1 in validation set')
# parser.add_argument('--hr2_val', default='CDD/val/time2', type=str, help='hr image at t2 in validation set')
# parser.add_argument('--lab_val', default='CDD/val/label', type=str, help='label image in validation set')

# # #test
parser.add_argument('--hr1_test', default='CDdata/LEVIR/test/time1', type=str, help='hr image at t1 in validation set')
parser.add_argument('--hr2_test', default='CDdata/LEVIR/test/time2', type=str, help='hr image at t2 in validation set')
parser.add_argument('--lab_test', default='CDdata/LEVIR/test/label', type=str, help='label image in vn validation set')


# network saving and loading parameters
parser.add_argument('--model_dir', default='epoch/levircd/samcd/', type=str, help='save path for CD model ')
parser.add_argument('--sr_dir', default='epochs/X4.00/SR/', type=str, help='save path for Generator')
parser.add_argument('--sta_dir', default='statistics/CDD_4x.csv', type=str, help='statistics save path')
