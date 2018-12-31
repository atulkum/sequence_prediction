import torch
import torch.nn.functional as F

def check_ignore_pads_in_loss():
    logits = torch.randn(2, 3, 4, requires_grad=True)
    y = torch.empty(2, 3, dtype=torch.long).random_(4)
    y[0][1:3] = -1
    log_smx = F.log_softmax(logits, dim=2)
    loss = F.nll_loss(log_smx.transpose(1, 2), y, ignore_index=-1, reduction='none')
    print (y)
    print (loss)

    assert loss[0][1] == 0.0 and loss[0][2] == 0.0

    print (loss.sum(dim=1))
    s_len = torch.Tensor([1, 3])
    loss = loss.sum(dim=1) / s_len
    print(loss)
    loss = loss.mean()
    print(loss)

if __name__ == '__main__':
    check_ignore_pads_in_loss()


