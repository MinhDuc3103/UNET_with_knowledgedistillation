import torch
import torch.nn.functional as F
from utils.dice_score import dice_loss
#NOTE: all verifications for the size is left

T = 1
alpha = 0.5

def loss_fn_kd(student_output, teacher_output, gt ):
    '''student_output = student_output.round() 
    student_output[student_output<0] = 0
    gt = torch.clamp(gt, min = 0, max = 1)
    teacher_output = torch.clamp(teacher_output, min = 0, max = 1)'''
    student_loss = general_loss(student_output, gt)
    #student_output = student_output.clamp(min = 1, max = 3)
    #teacher_output = teacher_output.clamp(min = 1, max = 3)

    kd_loss = pixel_wise_loss(student_output, teacher_output)
    #not sure about using T, also check KLD
    loss = (student_loss*(1-alpha) + (kd_loss)*(alpha)) # as per structured KD paper
    return loss


def general_loss(student_output, gt):
    #use torch.nn.CrossENtropyLoss()
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(student_output, gt)
    loss += dice_loss(
        F.softmax(student_output, dim=1).float(),
        F.one_hot(gt, 3).permute(0, 3, 1, 2).float(),
        multiclass=True
    )
    return loss

def pixel_wise_loss(student_output, teacher_output):
    N,C,W,H = student_output.shape

    #what would happen if we use softmax?
    pred_T = torch.sigmoid(teacher_output/T)
    pred_S = torch.sigmoid(student_output/T).log()

    #criterion = torch.nn.KLDivLoss(reduction = 'batchmean')
    #KLDloss = - criterion(pred_S, pred_T)
    #TODO: map this to KLDL
    #KDloss = - sum(p * log (p/q)) ---> refer notes page 15 - 16 
    #Pixelwise loss = sum(-p*logq)
    #KLDiv = relative entropy
    pixelwise_loss = (- pred_T * pred_S)
    return  torch.sum(pixelwise_loss) / (W*H)
    #return F.l1_loss(student_output, teacher_output)
