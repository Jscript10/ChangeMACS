from tqdm import tqdm
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
from configures import parser
#from model2.dsamnet import DSAMNet as UNet
from data_utils import LoadDatasetFromFolder_CD, calMetric_iou,Compute_Precision,Compute_Recall

if __name__ == "__main__":
    args = parser.parse_args()

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet().to(device, dtype=torch.float)
    path = 'epoch/levircd/our/netCD_epoch_149_pre_0.9003_rec_0.9267_f1_0.9133_iou_0.8405_OA_0.9866_KC_0.8717.pth'  # the path of the model2
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    test_set = LoadDatasetFromFolder_CD(args, args.hr1_test, args.hr2_test, args.lab_test)
    test_loader = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=10, shuffle=False)

    with torch.no_grad():
        val_bar = tqdm(test_loader)
        inter, unin = 0, 0
        inter1, unin1 = 0, 0
        inter2, unin2 = 0, 0
        inter3, unin3 = 0, 0
        valing_results = {'PRE':0,'REC': 0, 'F1':0, 'batch_sizes': 0, 'IoU': 0}
        for hr_img1, hr_img2, label in val_bar:
            valing_results['batch_sizes'] += args.val_batchsize
            hr_img1 = hr_img1.to(device, dtype=torch.float)
            hr_img2 = hr_img2.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float)
            label = torch.argmax(label, 1).unsqueeze(1).float()
            masks = net(hr_img1, hr_img2)
            gt_value = (label > 0).float()
            prob = (masks > 1).float()
            prob = prob.cpu().detach().numpy()
            gt_value = gt_value.cpu().detach().numpy()
            gt_value = np.squeeze(gt_value)
            result = np.squeeze(prob)
            intr, unn = calMetric_iou(result, gt_value)
            inter = inter + intr
            unin = unin + unn
            int1, unn1 = Compute_Precision(result, gt_value)
            inter1 = inter1 + int1
            unin1 = unin1 + unn1
            int2, unn2 = Compute_Recall(result, gt_value)
            inter2 = inter2 + int2
            unin2 = unin2 + unn2

            # loss for current batch before optimization

            valing_results['REC'] = (inter2 * 1.0 / unin2)
            valing_results['PRE'] = (inter1 * 1.0 / unin1)
            valing_results['F1'] = (2 * (valing_results['PRE'] * valing_results['REC']) / (valing_results['PRE'] + valing_results['REC']))
            valing_results['IoU'] = (inter * 1.0 / unin)
            val_bar.set_description(
                desc=' IoU:%.4f' % (
                    valing_results['IoU'])
            )

    val_loss = valing_results['IoU']
    print(valing_results)
