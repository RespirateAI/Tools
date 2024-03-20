import torch

torch.multiprocessing.set_sharing_strategy("file_system")
import argparse
import json
import os
from torch.utils.tensorboard import SummaryWriter
import random
from Model.Attention2 import Attention_Gated as Attention
from Model.Attention2 import Attention_with_Classifier
from utils import get_cam_1d
import torch.nn.functional as F
from Model.network import Classifier_1fc, DimReduction
import numpy as np
from utils import eval_metric

from LungData import LungDataset, custom_collate
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="abc")
testMask_dir = ""  ## Point to the Camelyon test set mask location

parser.add_argument("--name", default="abc", type=str)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--isPar", default=False, type=bool)
parser.add_argument("--log_dir", default="./debug_log", type=str)  ## log file path
parser.add_argument("--train_show_freq", default=40, type=int)
parser.add_argument("--droprate", default="0", type=float)
parser.add_argument("--droprate_2", default="0", type=float)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--batch_size_v", default=1, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--num_cls", default=2, type=int)
parser.add_argument("--numGroup", default=4, type=int)
parser.add_argument("--total_instance", default=4, type=int)
parser.add_argument("--numGroup_test", default=4, type=int)
parser.add_argument("--total_instance_test", default=4, type=int)
parser.add_argument("--mDim", default=4096, type=int)
parser.add_argument("--grad_clipping", default=5, type=float)
parser.add_argument("--isSaveModel", action="store_false")
parser.add_argument("--debug_DATA_dir", default="", type=str)
parser.add_argument("--numLayer_Res", default=0, type=int)
parser.add_argument("--temperature", default=1, type=float)
parser.add_argument("--num_MeanInference", default=1, type=int)
parser.add_argument("--distill_type", default="AFS", type=str)  ## MaxMinS, MaxS, AFS

torch.manual_seed(32)
torch.cuda.manual_seed(32)
np.random.seed(32)
random.seed(32)


def main():
    params = parser.parse_args()
    writer = SummaryWriter(os.path.join(params.log_dir, "LOG", params.name))

    classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(
        params.device
    )
    dimReduction = DimReduction(4096, params.mDim, numLayer_Res=params.numLayer_Res).to(
        params.device
    )
    attention = Attention(params.mDim).to(params.device)
    attCls = Attention_with_Classifier(
        L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2
    ).to(params.device)

    if params.isPar:
        classifier = torch.nn.DataParallel(classifier)
        dimReduction = torch.nn.DataParallel(dimReduction)
        attention = torch.nn.DataParallel(attention)
        attCls = torch.nn.DataParallel(attCls)

    # Load the saved state dictionaries
    tsave_dict = torch.load(
        os.path.join(params.log_dir, "NEW_COMB", "random_binary_resnet_32_comb_train.pth")
    )

    # Load the state dictionaries into the respective model components
    classifier.load_state_dict(tsave_dict["classifier"])
    dimReduction.load_state_dict(tsave_dict["dimReduction"])
    attention.load_state_dict(tsave_dict["attention"])
    attCls.load_state_dict(tsave_dict["att_classifier"])

    ce_cri = torch.nn.CrossEntropyLoss(reduction="none").to(params.device)

    if not os.path.exists(params.log_dir):
        os.makedirs(params.log_dir)
    log_dir = os.path.join(params.log_dir, "log.txt")
    # save_dir = os.path.join(params.log_dir, "best_model_lung2.pth")
    z = vars(params).copy()
    with open(log_dir, "a") as f:
        f.write(json.dumps(z))
    log_file = open(log_dir, "a")

    trainable_parameters = []
    trainable_parameters += list(classifier.parameters())
    trainable_parameters += list(attention.parameters())
    trainable_parameters += list(dimReduction.parameters())

    auc_val = test_attention_DTFD_preFeat_MultipleMean(
        classifier=classifier,
        dimReduction=dimReduction,
        attention=attention,
        UClassifier=attCls,
        criterion=ce_cri,
        params=params,
        f_log=log_file,
        writer=writer,
        numGroup=params.numGroup_test,
        total_instance=params.total_instance_test,
        distill=params.distill_type,
    )


def test_attention_DTFD_preFeat_MultipleMean(
    classifier,
    dimReduction,
    attention,
    UClassifier,
    criterion=None,
    params=None,
    f_log=None,
    writer=None,
    numGroup=3,
    total_instance=3,
    distill="MaxMinS",
):

    data = LungDataset(
        "/media/wenuka/New Volume-G/01.FYP/Dataset/512_splitted/test"
    )
    dataloader = DataLoader(data, params.batch_size, collate_fn=custom_collate)

    classifier.eval()
    dimReduction.eval()
    attention.eval()
    UClassifier.eval()

    # SlideNames, FeatLists, Label = mDATA_list
    instance_per_group = total_instance // numGroup

    test_loss0 = AverageMeter()
    test_loss1 = AverageMeter()

    gPred_0 = torch.FloatTensor().to(params.device)
    gt_0 = torch.LongTensor().to(params.device)
    gPred_1 = torch.FloatTensor().to(params.device)
    gt_1 = torch.LongTensor().to(params.device)

    names = []

    attention_weights_dic = {}

    with torch.no_grad():

        numSlides = len(data)
        numIter = numSlides // params.batch_size_v
        tIDX = list(range(numSlides))

        # for idx in range(numIter):
        for batch in dataloader:

            # tidx_slide = tIDX[
            #     idx * params.batch_size_v : (idx + 1) * params.batch_size_v
            # ]
            slide_names_batch = batch["slide_name"]
            features_batch = batch["features"]
            labels_batch = batch["label"]
            # slide_names = [SlideNames[sst] for sst in tidx_slide]
            # tlabel = [Label[sst] for sst in tidx_slide]
            label_tensor = torch.LongTensor(labels_batch).to(params.device)
            batch_feat = [torch.tensor(f).to(params.device) for f in features_batch]

            # print(f"Batch feat size: {batch_feat.size()}")

            for tidx, tfeat in enumerate(batch_feat):
                tslideName = slide_names_batch[tidx]
                tslideLabel = label_tensor[tidx].unsqueeze(0)
                midfeat = dimReduction(tfeat)

                # print(f"Input {tfeat.shape}")
                AA = attention(midfeat, isNorm=False).squeeze(0)  ## N
                # print(f"Attention weights {AA.shape}")
                # attention_weights[tslideName] = AA
                attention_weights_dic[tslideName] = {}

                allSlide_pred_softmax = []

                for jj in range(params.num_MeanInference):

                    feat_index = list(range(tfeat.shape[0]))
                    random.shuffle(feat_index)
                    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
                    index_chunk_list = [sst.tolist() for sst in index_chunk_list]

                    slide_d_feat = []
                    slide_sub_preds = []
                    slide_sub_labels = []

                    for t, tindex in enumerate(index_chunk_list):
                        slide_sub_labels.append(tslideLabel)
                        idx_tensor = torch.LongTensor(tindex).to(params.device)
                        # print(idx_tensor)
                        tmidFeat = tfeat.index_select(dim=0, index=idx_tensor)
                        # print(f"tmidfeat shape: {tmidFeat.shape}")

                        tAA = AA.index_select(dim=0, index=idx_tensor)
                        # print(f"TAA shape: {tAA.shape}")
                        # print(f"TAA: {tAA}")
                        tAA = torch.softmax(tAA, dim=0)
                        # print(f"TAA shape after softmax: {tAA.shape}")
                        # print(f"TAA after softmax: {tAA}")
                        attention_weights_dic = create_dictionary(
                            attention_weights_dic, tslideName, tindex, t, tAA
                        )

                        tattFeats = torch.einsum("ns,n->ns", tmidFeat, tAA)  ### n x fs
                        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(
                            0
                        )  ## 1 x fs

                        tPredict = classifier(tattFeat_tensor)  ### 1 x 2
                        slide_sub_preds.append(tPredict)

                        patch_pred_logits = get_cam_1d(
                            classifier, tattFeats.unsqueeze(0)
                        ).squeeze(
                            0
                        )  ###  cls x n
                        patch_pred_logits = torch.transpose(
                            patch_pred_logits, 0, 1
                        )  ## n x cls
                        patch_pred_softmax = torch.softmax(
                            patch_pred_logits, dim=1
                        )  ## n x cls

                        _, sort_idx = torch.sort(
                            patch_pred_softmax[:, -1], descending=True
                        )

                        if distill == "MaxMinS":
                            topk_idx_max = sort_idx[:instance_per_group].long()
                            topk_idx_min = sort_idx[-instance_per_group:].long()
                            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                            d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                            slide_d_feat.append(d_inst_feat)
                        elif distill == "MaxS":
                            topk_idx_max = sort_idx[:instance_per_group].long()
                            topk_idx = topk_idx_max
                            d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                            slide_d_feat.append(d_inst_feat)
                        elif distill == "AFS":
                            slide_d_feat.append(tattFeat_tensor)

                    slide_d_feat = torch.cat(slide_d_feat, dim=0)
                    # print(f"Slide d feat shape: {slide_d_feat.shape}")
                    slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
                    slide_sub_labels = torch.cat(slide_sub_labels, dim=0)

                    gPred_0 = torch.cat([gPred_0, slide_sub_preds], dim=0)
                    gt_0 = torch.cat([gt_0, slide_sub_labels], dim=0)
                    loss0 = criterion(slide_sub_preds, slide_sub_labels).mean()
                    test_loss0.update(loss0.item(), numGroup)

                    gSlidePred, att = UClassifier(slide_d_feat)
                    # print(f"Att shape: {att.shape}")
                    # print(f"Att: {att}")
                    attention_weights_dic[tslideName]["tier 2"] = att
                    allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))

                allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
                allSlide_pred_softmax = torch.mean(
                    allSlide_pred_softmax, dim=0
                ).unsqueeze(0)
                gPred_1 = torch.cat([gPred_1, allSlide_pred_softmax], dim=0)
                gt_1 = torch.cat([gt_1, tslideLabel], dim=0)

                loss1 = F.nll_loss(allSlide_pred_softmax, tslideLabel)
                test_loss1.update(loss1.item(), 1)

                names.append(tslideName)

    gPred_0 = torch.softmax(gPred_0, dim=1)
    gPred_0 = gPred_0[:, -1]
    gPred_1 = gPred_1[:, -1]

    # print(gt_1)
    # print(gPred_1)

    macc_0, mprec_0, mrecal_0, mspec_0, mF1_0, auc_0, threshold0 = eval_metric(
        gPred_0, gt_0
    )
    macc_1, mprec_1, mrecal_1, mspec_1, mF1_1, auc_1, threshold1 = eval_metric(
        gPred_1, gt_1
    )

    print_log(
        f"  First-Tier acc {macc_0}, precision {mprec_0}, recall {mrecal_0}, specificity {mspec_0}, F1 {mF1_0}, AUC {auc_0}",
        f_log,
    )
    print_log(
        f"  Second-Tier acc {macc_1}, precision {mprec_1}, recall {mrecal_1}, specificity {mspec_1}, F1 {mF1_1}, AUC {auc_1}",
        f_log,
    )

    gPred_1[gPred_1 > threshold1] = 1
    gPred_1[gPred_1 <= threshold1] = 0

    gt_1[gt_1 > 0.5] = 1
    gt_1[gt_1 <= 0.5] = 0

    combined_data = list(zip(gt_1, gPred_1, names))

    # for idx, (ground_truth, prediction, name) in enumerate(combined_data):
    #     print(
    #         f"Index: {idx}, Name: {name}, Ground Truth: {ground_truth.item()}, Prediction: {prediction.item()}"
    #     )

    # numpy_data_dict = {
    #     key: value.cpu().numpy() for key, value in attention_weights.items()
    # }

    # Save the dictionary to a .npy file
    np.save(
        "./Attention/attention_weights_resnet_random_binary_resnet_32_comb_train_new.npy",
        attention_weights_dic,
    )

    # with open("test_resnet_random_binary_resnet_32_comb_train.csv", "w") as f:
    #     f.write("Name,Ground Truth,Resnet_Prediction\n")
    #     for ground_truth, prediction, name in combined_data:
    #         f.write(f"{name},{ground_truth.item()},{prediction.item()}\n")


def create_dictionary(dic, imgName, idx_list, idx_list_index, attention_weights):
    for i, val in enumerate(idx_list):
        dic[imgName][val] = (attention_weights[i], idx_list_index)
    return dic


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_log(tstr, f):
    # with open(dir, 'a') as f:
    f.write("\n")
    f.write(tstr)
    print(tstr)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)

    main()
