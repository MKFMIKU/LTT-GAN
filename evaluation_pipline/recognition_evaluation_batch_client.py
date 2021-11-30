import os.path as osp
import sys
import os
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
path = osp.join(this_dir, 'GPEN')
path = osp.join(this_dir, 'GPEN/retinaface')

add_path(path)

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import glob
import cv2
import torch
from torch.nn import functional as F
from gfpgan.archs.arcface_arch import ResNetArcFace
from copy import deepcopy
from basicsr.utils import img2tensor, tensor2img
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np
from GPEN.retinaface.retinaface_detection import RetinaFaceDetection
from GPEN.align_faces import warp_and_crop_face, get_reference_facial_points
from lpips_pytorch import lpips
from niqe import niqe

def main():
    def find_nearest_idx(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    # def get_face(img,resp):
    #     img = deepcopy(img)
    #     #y1, x1, y2, x2 = resp # resp['face_1']['facial_area']
    #
    #     x, y, w, h =resp
    #     face = img[int(y):int(y+h), int(x):int(x+w)]
    #
    #     factor_0 = 224 / face.shape[0]
    #     factor_1 = 224 / face.shape[1]
    #     factor = min(factor_0, factor_1)
    #
    #     dsize = (int(face.shape[1] * factor), int(face.shape[0] * factor))
    #     img = cv2.resize(face, dsize)
    #     diff_0 = 224 - img.shape[0]
    #     diff_1 = 224 - img.shape[1]
    #     img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)),
    #                  'constant')
    #     return img

    def imwrite(img, file_path, params=None, auto_mkdir=True):
        """Write image to file.

        Args:
            img (ndarray): Image array to be written.
            file_path (str): Image file path.
            params (None or list): Same as opencv's :func:`imwrite` interface.
            auto_mkdir (bool): If the parent folder of `file_path` does not exist,
                whether to create it automatically.

        Returns:
            bool: Successful or not.
        """
        if auto_mkdir:
            dir_name = os.path.abspath(os.path.dirname(file_path))
            os.makedirs(dir_name, exist_ok=True)
        ok = cv2.imwrite(file_path, img, params)
        if not ok:
            raise IOError('Failed in writing images.')
    def gray_resize_for_identity(out, size=128):
        out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
        out_gray = out_gray.unsqueeze(1)
        out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
        return out_gray

    def load_network(net, load_path, strict=True, param_key='params'):
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        net.load_state_dict(load_net, strict=strict)

    use_arcface=True
    if use_arcface:
        arcface_resnet18 = ResNetArcFace(block='IRBlock', layers=[2, 2, 2, 2], use_se=False)
        load_network(arcface_resnet18, './fedbase/experiments/pretrained_models/arcface_resnet18.pth')
    else:
        arcface_resnet18=VGGFace()
        arcface_resnet18.load_state_dict(torch.load(os.path.join('/home/yiqunm2/workspace/fl/fedbase/experiments/pretrained_models/vggface.pth')))

    arcface_resnet18.cuda()
    arcface_resnet18.eval()

    client = 1
    itr = 80000
    detect_face = True
    exp_name = 'train_GFPGANv2_512_non_iid_n8_a03_FL_prox_silo_200_mixed_avgfreq_ddp_atsuperAPT' # 'train_GFPGANv2_512_non_iid_n8_a03_FL_avg_silo_200_single_client_ddp_c3'
    gt_dir = './full_set_mixed_transfered/VIS'
    ret_images = glob.glob('./fedbase/experiments/{}/visualization/c{}_*/*{}.png'.format(exp_name,client,itr))
    ret_images.sort()
    print()

    spoort_metric_verification =['DEG', 'AUC','EER','VR@FAR=1%','VR@FAR=1%','Top1','Top3']
    spoort_metric_img_quality = ['PSNR', 'SSIM', 'LPIPS', 'NIQE']
    metric_dict= {}
    degs = []
    degs_face = []
    id_th_list =[]
    ref_img_list = []
    ret_img_list = []
    psnrs = []
    ssims = []
    lpipss = []
    niqes = []
    if detect_face:
        # retinaface
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        size = 128
        reference_5pts = get_reference_facial_points((size, size), inner_padding_factor, outer_padding, default_square)
        facedetector = RetinaFaceDetection('./GPEN')
        ref_img_face_list = []
        ret_img_face_list = []
        degs_face = []
    # FOR COSINE DEGREE ONLY!
    for idx in tqdm(range(len(ret_images))):
        id_th_list.append(int(os.path.basename(ret_images[idx])[:3]))
        # to compatible with deepface apt, save two img separately
        fname = ret_images[idx]
        combine_img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        ref_img = combine_img[:,512:,:]
        ret_img = combine_img[:,:512,:]
        # for pre, gt are saved in separate file
        # gt_path = os.path.join(gt_dir,'_'.join(os.path.basename(ret_images[idx]).split('_')[:-1])+ '.png')
        # ref_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        psnrs.append(calculate_psnr(ret_img,ref_img,crop_border=0))
        ssims.append(calculate_ssim(ret_img,ref_img,crop_border=0))

        if detect_face:
            facebs, landms = facedetector.detect(ref_img[:,:,::-1])
            if landms.shape[0] == 0 or 'LD' in fname:
                # face is not detected, we just give img without face detection
                ref_img_face = ref_img
                ret_img_face = ret_img
                print('face is not detected', fname)
            else:
                for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
                    facial5points = np.reshape(facial5points, (2, 5))

                    ref_img_face, tfm_inv = warp_and_crop_face(ref_img, facial5points, reference_pts=reference_5pts,crop_size=(size, size))
                    ret_img_face, tfm_inv = warp_and_crop_face(ret_img, facial5points, reference_pts=reference_5pts,crop_size=(size, size))
                    # only detect one face
                    break
            # plt.imshow(ref_img_face)
            # plt.show()
            # plt.imshow(ret_img_face)
            # plt.show()
            ref_img_face = img2tensor(ref_img_face / 255., bgr2rgb=True, float32=True)
            ref_img_face = ref_img_face.unsqueeze(0).cuda()
            ref_img_face = normalize(ref_img_face, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            ref_img_face = gray_resize_for_identity(ref_img_face)

            ret_img_face = img2tensor(ret_img_face / 255., bgr2rgb=True, float32=True)
            ret_img_face = ret_img_face.unsqueeze(0).cuda()
            ret_img_face = normalize(ret_img_face, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            ret_img_face = gray_resize_for_identity(ret_img_face)

        ref_img = img2tensor(ref_img / 255., bgr2rgb=True, float32=True)
        ref_img_before_norm = ref_img.unsqueeze(0).cuda()
        ref_img_for_img_quality = gray_resize_for_identity(ref_img_before_norm,512)
        ref_img = normalize(ref_img_before_norm, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ref_img = gray_resize_for_identity(ref_img)

        ret_img = img2tensor(ret_img / 255., bgr2rgb=True, float32=True)
        ret_img_before_norm = ret_img.unsqueeze(0).cuda()
        ret_img_for_img_quality = gray_resize_for_identity(ret_img_before_norm, 512)
        ret_img = normalize(ret_img_before_norm, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ret_img = gray_resize_for_identity(ret_img)


        lpipss.append(lpips(ref_img_for_img_quality * 2 - 1, ret_img_for_img_quality * 2 - 1, net_type="alex", version="0.1").item())
        niqes.append(niqe(ret_img_for_img_quality.cpu().squeeze().numpy()))
        with torch.no_grad():
            ret_img = arcface_resnet18(ret_img)
            ref_img = arcface_resnet18(ref_img)
            if detect_face:
                ret_img_face = arcface_resnet18(ret_img_face)
                ref_img_face = arcface_resnet18(ref_img_face)
        ref_img_list.append(ref_img)
        ret_img_list.append(ret_img)
        deg = F.cosine_similarity(ref_img, ret_img).item()
        degs.append(deg)

        if detect_face:
            ref_img_face_list.append(ref_img_face)
            ret_img_face_list.append(ret_img_face)
            deg = F.cosine_similarity(ref_img_face, ret_img_face).item()
            degs_face.append(deg)

    metric_dict['DEG'] = sum(degs) / len(degs)
    metric_dict['PSNR'] = sum(psnrs) / len(psnrs)
    metric_dict['SSIM'] = sum(ssims) / len(ssims)
    metric_dict['LPIPS'] = sum(lpipss) / len(lpipss)
    metric_dict['NIQE'] = sum(niqes) / len(niqes)
    if detect_face:
        metric_dict['DEG_face'] = sum(degs_face) / len(degs_face)
    # select gallery img
    if client>4:
        # for our own dataset, just choose the first image of a subject as gallery
        tmp = list(set([os.path.basename(i)[:3] for i in ret_images]))
        ref_images = []
        for i in ret_images:
            if os.path.basename(i)[:3] in tmp:
                ref_images.append(i)
                tmp.remove(os.path.basename(i)[:3])
    else:
        # for TH-VIS, just select the NN image of a subject as gallery
        ref_images = [i for i in ret_images if 'NN' in i]

    ref_images.sort()
    # recognition 
    ref_img_list=[]
    ref_img_face_list = []
    id_ref_list = []
    for idx in tqdm(range(len(ref_images))):
        # read image with nn pose
        ref_id = int(os.path.basename(ref_images[idx])[:3])
        id_ref_list.append(ref_id)
    
        combine_img = cv2.imread(ref_images[idx], cv2.IMREAD_UNCHANGED)
        ref_img = combine_img[:,512:,:]
        if detect_face:
            facebs, landms = facedetector.detect(ref_img[:,:,::-1])
            for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
                facial5points = np.reshape(facial5points, (2, 5))
                ref_img_face, tfm_inv = warp_and_crop_face(ref_img, facial5points, reference_pts=reference_5pts,crop_size=(size, size))
                # only detect one face
                break
            ref_img_face = img2tensor(ref_img_face / 255., bgr2rgb=True, float32=True)
            ref_img_face = ref_img_face.unsqueeze(0).cuda()
            ref_img_face = normalize(ref_img_face, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            ref_img_face = gray_resize_for_identity(ref_img_face)

        ref_face_img_th = img2tensor(ref_img / 255., bgr2rgb=True, float32=True)
        ref_face_img_th = ref_face_img_th.unsqueeze(0).cuda()
        ref_face_img_th = normalize(ref_face_img_th, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ref_face_img_th = gray_resize_for_identity(ref_face_img_th)

        with torch.no_grad():
            ref_face_img_th = arcface_resnet18(ref_face_img_th)
            if detect_face:
                ref_face_img_face_th = arcface_resnet18(ref_img_face)
                ref_img_face_list.append(ref_face_img_face_th)
        ref_img_list.append(ref_face_img_th)
    ret_imgs = torch.cat(ret_img_list).cpu().numpy()
    ref_imgs = torch.cat(ref_img_list).cpu().numpy()
    # For AUC/ ERR /. the process is deferent
    # A set of results with expression (probes)
    # Matches a set Netural Pose (gallery)
    if detect_face:
        ret_imgs_face = torch.cat(ret_img_face_list).cpu().numpy()
        ref_imgs_face = torch.cat(ref_img_face_list).cpu().numpy()
        similarity_matrix_face = np.asarray(cosine_similarity(ret_imgs_face, ref_imgs_face))
        gt_matrix_face = np.zeros([similarity_matrix_face.shape[0], similarity_matrix_face.shape[1]])

        for i in range(similarity_matrix_face.shape[0]):
            for j in range(similarity_matrix_face.shape[1]):
                if id_th_list[i] == id_ref_list[j]:
                    gt_matrix_face[i, j] = 1
                else:
                    gt_matrix_face[i, j] = -1
        fpr, tpr, thresholds = roc_curve(gt_matrix_face.reshape(-1), similarity_matrix_face.reshape(-1))
        roc_auc = auc(fpr, tpr)

        # find negative pairs
        negative_pair_scores = similarity_matrix_face.reshape(-1)[gt_matrix_face.reshape(-1) == -1.0]
        n_negative_pair = len(negative_pair_scores)
        # Sort in decending order
        negative_pair_scores = np.sort(negative_pair_scores)[::-1]
        # VR@FAR=1%
        thresh_1 = negative_pair_scores[round(n_negative_pair * 0.01)]
        idx_1 = find_nearest_idx(thresholds, thresh_1)
        #print('face VR@FAR=1% :', tpr[idx_1])
        metric_dict['VR@FAR=1%_face'] = tpr[idx_1]
        # VR@FAR=0.1%
        thresh_01 = negative_pair_scores[round(n_negative_pair * 0.001)]
        idx_01 = find_nearest_idx(thresholds, thresh_01)
        #print('face VR@FAR=0.1% :', tpr[idx_01])
        metric_dict['VR@FAR=0.1%_face'] = tpr[idx_01]

        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)

        # print('face ROC AUC: %f ' % (roc_auc))
        # print('face ROC EER: %f ' % (eer))
        metric_dict['AUC_face'] = roc_auc
        metric_dict['EER_face'] = eer

    similarity_matrix = np.asarray(cosine_similarity(ret_imgs,ref_imgs))
    gt_matrix = np.zeros([similarity_matrix.shape[0], similarity_matrix.shape[1]])
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            if id_th_list[i] == id_ref_list[j]:
                gt_matrix[i, j] = 1
            else:
                gt_matrix[i, j] = -1
    fpr, tpr, thresholds = roc_curve(gt_matrix.reshape(-1), similarity_matrix.reshape(-1))
    roc_auc = auc(fpr, tpr)

    # find negative pairs
    negative_pair_scores = similarity_matrix.reshape(-1)[gt_matrix.reshape(-1)==-1.0]
    n_negative_pair = len(negative_pair_scores)
    # Sort in decending order
    negative_pair_scores = np.sort(negative_pair_scores)[::-1]
    # VR@FAR=1%
    thresh_1 = negative_pair_scores[round(n_negative_pair*0.01)]
    idx_1 = find_nearest_idx(thresholds,thresh_1)
    #print('VR@FAR=1% :', tpr[idx_1])
    metric_dict['VR@FAR=1%'] = tpr[idx_1]

    # VR@FAR=0.1%
    thresh_01 = negative_pair_scores[round(n_negative_pair*0.001)]
    idx_01 = find_nearest_idx(thresholds,thresh_01)
    #print('VR@FAR=0.1% :', tpr[idx_01])
    metric_dict['VR@FAR=0.1%'] = tpr[idx_01]

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)

    # print('ROC AUC: %f ' % (roc_auc))
    # print('ROC EER: %f ' % (eer))
    metric_dict['AUC'] = roc_auc
    metric_dict['EER'] = eer
    if detect_face:
        top_inds_face = torch.argsort(-torch.Tensor(similarity_matrix_face))
        # top3
        correct_num = 0
        for i in range(ret_imgs.shape[0]):
            j = top_inds_face[i, 0:3]
            j_id = [id_ref_list[k] for k in j]
            if id_th_list[i] in j_id:
                correct_num += 1
        #print("top3 = {}".format(correct_num / ret_imgs.shape[0]))
        metric_dict['Top3_face'] = correct_num / ret_imgs.shape[0]
        # top1 for face
        correct_num = 0
        for i in range(ret_imgs.shape[0]):

            j = top_inds_face[i, 0]
            if id_ref_list[j] == id_th_list[i]:
                correct_num += 1
        #print("face top1 = {}".format(correct_num / ret_imgs.shape[0]))
        metric_dict['Top1_face'] = correct_num / ret_imgs.shape[0]

    top_inds = torch.argsort(-torch.Tensor(similarity_matrix))

    # top3
    correct_num = 0
    for i in range(ret_imgs.shape[0]):
        j = top_inds[i, 0:3]
        j_id = [id_ref_list[k] for k in j]
        if id_th_list[i] in j_id:
            correct_num += 1
    #print("face top1 = {}".format(correct_num / ret_imgs.shape[0]))
    metric_dict['Top3'] = correct_num / ret_imgs.shape[0]
    #top1
    correct_num = 0
    for i in range(ret_imgs.shape[0]):
        j = top_inds[i, 0]
        if id_ref_list[j] == id_th_list[i]:
            correct_num += 1
    #print("top1 = {}".format(correct_num / ret_imgs.shape[0]))
    metric_dict['Top1'] = correct_num / ret_imgs.shape[0]

    # print final metric
    print('For verification metric client: {} -----------------'.format(client))
    for m in spoort_metric_verification:
        if detect_face:
            print(m + ' :', round(metric_dict[m], 4), m+'_face' + ' :', round(metric_dict[m+'_face'], 4))
        else:
            print(m + ' :',round(metric_dict[m],4))
    print('For image quality metric client: {} -----------------'.format(client))
    for m in spoort_metric_img_quality:
        print(m + ' :',round(metric_dict[m],4))



if __name__ == '__main__':
    main()
