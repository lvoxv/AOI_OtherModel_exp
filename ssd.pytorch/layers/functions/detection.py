import torch
from torch.autograd import Function
from ..box_utils import decode, nms
from data import voc as cfg


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    @staticmethod
    def forward(ctx, loc_data, conf_data, prior_data, num_classes=21, bkg_label=0, 
                top_k=200, conf_thresh=0.01, nms_thresh=0.45):
        """
        Args:
            ctx: autograd context
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
            num_classes, bkg_label, top_k, conf_thresh, nms_thresh: 检测参数
        """
        # 将参数保存到上下文中，以便在backward中使用
        ctx.num_classes = num_classes
        ctx.background_label = bkg_label
        ctx.top_k = top_k
        ctx.nms_thresh = nms_thresh
        ctx.conf_thresh = conf_thresh
        ctx.variance = cfg['variance']
        
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, num_classes, top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, ctx.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, num_classes):
                c_mask = conf_scores[cl].gt(conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, nms_thresh, top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # 在推理阶段通常不需要反向传播，所以返回None
        return None, None, None, None, None, None, None, None