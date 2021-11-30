First Install package:
pip install git+https://github.com/S-aiueo32/lpips-pytorch.git

Move all files to same root dir as fedbase, including:

recognition_evaluation_batch_client.py (the main function)
GPEN (for face detection)
modelparameters.mat (for niqe)
niqe.py (for niqe)

spoort_metric_verification:'DEG', 'AUC','EER','VR@FAR=1%','VR@FAR=1%','Top1','Top3']
if you enable face_detection, all verification metric will produce both results with and without face detection.

spoort_metric_img_quality:'PSNR', 'SSIM', 'LPIPS', 'NIQE']
example output:
    For verification metric client: 1 -----------------
    DEG : 0.417 DEG_face : 0.4503
    AUC : 0.8225 AUC_face : 0.8284
    EER : 0.2857 EER_face : 0.254
    VR@FAR=1% : 0.3175 VR@FAR=1%_face : 0.2698
    VR@FAR=1% : 0.3175 VR@FAR=1%_face : 0.2698
    Top1 : 0.6667 Top1_face : 0.746
    Top3 : 1.0 Top3_face : 1.0
    For image quality metric client: 1 -----------------
    PSNR : 16.9574
    SSIM : 0.6827
    LPIPS : 0.2898
    NIQE : 0.6252