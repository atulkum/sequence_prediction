import torch
import torch.nn.functional as F
import numpy as np

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
    #check_ignore_pads_in_loss()
    ''' 
    a_ = np.random.randint(1, 100, 5)
    b_ = np.random.randint(1, 100, 5)

    ab = sorted([aa*bb for bb in b_ for aa in a_], reverse=True)

    print (ab[:5], a_, b_)

    import heapq
    a = sorted(a_, reverse=True)
    b = sorted(b_, reverse=True)

    pQueue = []
    heapq.heappush(pQueue, (-a[0]*b[0], 0, 0))
    topk = []
    for _ in range(5):
        v, ia, ib = heapq.heappop(pQueue)
        topk.append(-v)
        if ia + 1 < len(a):
            heapq.heappush(pQueue, (-a[ia + 1]*b[ib], ia+1, ib))
        if ib + 1 < len(b):
            heapq.heappush(pQueue, (-a[ia] * b[ib+1], ia, ib+1))

    print (topk)
    '''
    tx = np.random.randint(1, 100, size=(5,5))
    e = np.random.randint(1, 100, size=5)

    print (tx)
    print(e)
    print(np.expand_dims(e, -1) + tx)



