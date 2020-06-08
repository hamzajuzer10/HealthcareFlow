import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from pathway_los_prediction import models, datasets, utils, model_configuration
import os
import json

# Data parameters
data_folder = '..\\sample_datasets'  # folder with data files saved by create_input_dataset.py
model_name = 'cap_6_pathways_20_min_ward_freq'  # model name to save trained model

# Model parameters
emb_dim = 4  # dimension of pathway embeddings
decoder_dim = 4  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 1  # for data-loading;
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
best_score = 0.  # score right now
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
print_freq = 100  # print training/validation stats every __ batches
ward_map = None


def main():
    """
    Training and validation.
    """

    global best_score, epochs_since_improvement, start_epoch, model_name, ward_map, cat_emb_dim

    # Load model configuration
    config = model_configuration.generate_pathway_los_config()

    # Read word map
    ward_map_file = os.path.join(data_folder, 'WARD_MAP' + '.json')
    with open(ward_map_file, 'r') as j:
        ward_map = json.load(j)

    # Read embedding dimension
    with open(os.path.join(data_folder, 'EMB_DIMS' + '.json'), 'r') as j:
        cat_emb_dim = json.load(j)

    # Initialize
    encoder = models.Encoder(cat_emb_dim, len(config['pathway_los_continuous_features']))
    encoder_dim = encoder.get_output_dim()
    decoder = models.DecoderWithAttention(embed_dim=emb_dim,
                                          decoder_dim=decoder_dim,
                                          vocab_size=len(ward_map),
                                          encoder_dim=encoder_dim,
                                          dropout=dropout)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=decoder_lr)

    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(
        datasets.PathwayDataset(data_folder, 'TRAIN'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.PathwayDataset(data_folder, 'VAL'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            utils.adjust_learning_rate(decoder_optimizer, 0.8)
            utils.adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_score = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # Check if there was an improvement
        is_best = recent_score < best_score
        best_score = min(recent_score, best_score)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        utils.save_checkpoint(model_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                              decoder_optimizer, recent_score, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = utils.AverageMeter()  # forward prop. + back prop. time
    data_time = utils.AverageMeter()  # data loading time
    losses = utils.AverageMeter()  # loss (per word decoded)
    top5accs = utils.AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (cn_features, ct_features, pathway, p_len, los) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        cn_features = cn_features.to(device)
        ct_features = ct_features.to(device)
        pathway = pathway.to(device)
        p_len = p_len.to(device)
        los = los.to(device)

        # Forward prop.
        enc = encoder(cn_features, ct_features)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(enc, pathway, p_len)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            utils.clip_gradient(decoder_optimizer, grad_clip)
            utils.clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        encoder_optimizer.step()

        # Keep track of metrics
        top5 = utils.accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    encoder.eval()

    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top5accs = utils.AverageMeter()

    start = time.time()

    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (cn_features, ct_features, pathway, p_len, los) in enumerate(val_loader):

            # Move to device, if available
            cn_features = cn_features.to(device)
            ct_features = ct_features.to(device)
            pathway = pathway.to(device)
            p_len = p_len.to(device)
            los = los.to(device)

            # Forward prop.
            enc = encoder(cn_features, ct_features)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(enc, pathway, p_len)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = utils.accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}\n'.format(
                loss=losses,
                top5=top5accs))

    return losses.avg


if __name__ == '__main__':
    main()