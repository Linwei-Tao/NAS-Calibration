import torch
import torch.nn as nn
# Import metrics to compute
from Metrics.metrics import test_classification_net_logits
from Metrics.metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss


def get_logits_labels(data_loader, net):
    logits_list = []
    labels_list = []
    net.eval()
    with torch.no_grad():
        for data, label in data_loader:
            data = data.cuda()
            try:
                logits, _ = net(data)
            except:
                logits = net(data)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
    return logits, labels


def test_performance(test_queue, model):
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = ECELoss().cuda()
    adaece_criterion = AdaptiveECELoss().cuda()
    cece_criterion = ClasswiseECELoss().cuda()

    logits, labels = get_logits_labels(test_queue, model)
    conf_matrix, p_accuracy, _, _, _ = test_classification_net_logits(logits, labels)

    p_ece = ece_criterion(logits, labels).item()
    p_adaece = adaece_criterion(logits, labels).item()
    p_cece = cece_criterion(logits, labels).item()
    p_nll = nll_criterion(logits, labels).item()

    return p_ece, p_adaece, p_cece, p_nll
