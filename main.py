import torch

import utility
import data
import models
import loss2
from options.option5 import args
from trainers.trainerMS import Trainer
import os
import numpy as np
import random

checkpoint = utility.Checkpoint(args)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if checkpoint.ok:
    seed = 3407
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    #torch.manual_seed(seed)
    random.seed(seed)

    loader = data.Data(args)
    model = models.Model(args, checkpoint)
    print('# model parameters:',sum(param.numel() for param in model.parameters()))
    print(model)
    loss = loss2.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()
