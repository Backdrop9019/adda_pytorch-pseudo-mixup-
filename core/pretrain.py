import torch.nn as nn
import torch.optim as optim

import params
from utils import make_cuda, save_model, LabelSmoothingCrossEntropy,mixup_data
from random import *
import sys

def train_src(encoder, classifier, source_data_loader,target_data_loader,data_loader_eval):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()


    if params.usemixup:
    
        target_data_loader = list(target_data_loader)


    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))


    if params.labelsmoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing= params.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()


    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):

        for step, (images, labels) in enumerate(source_data_loader):

            # make images and labels variable
            images = make_cuda(images)
            labels = make_cuda(labels.squeeze_())
            # zero gradients for optimizer
            optimizer.zero_grad()

            # source , target   :  mixup
            if params.usemixup:
                images, lam = mixup_data(images,target_data_loader[randint(0, len(target_data_loader)-1)][0])


            # compute loss for critic
            preds = classifier(encoder(images))
            # print(f'images shape {images.shape}')
            # print(f'label shape {labels.shape}')
            # print(f'preds shape {preds.shape}')

            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()
            
            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(source_data_loader),
                              loss.item()))

        # eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_src(encoder, classifier, source_data_loader)
            print("eval", end= '\t')
            eval_src(encoder, classifier, data_loader_eval)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1))
            save_model(
                classifier, "ADDA-source-classifier-{}.pt".format(epoch + 1))

    # # save final model
    save_model(encoder, "ADDA-source-encoder-final.pt")
    save_model(classifier, "ADDA-source-classifier-final.pt")

    return encoder, classifier


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()


    # evaluate network
    for (images, labels) in data_loader:

        images = make_cuda(images)
        labels = make_cuda(labels)

        preds = classifier(encoder(images))
        loss += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum().item()

    loss /= len(data_loader)+1
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))