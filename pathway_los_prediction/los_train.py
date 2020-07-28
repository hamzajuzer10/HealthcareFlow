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
model_folder = '..\\pathway_los_prediction_models' # folder with the saved prediction models
model_name = 'cap_6_los_20_min_ward_freq'  # model name to save trained model

# Model parameters
emb_dim = 4  # dimension of pathway embeddings
decoder_dim = 6  # dimension of decoder RNN
dropout = 0.05
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
batch_size = 64
workers = 1  # for data-loading;
encoder_lr = 0.05  # learning rate for encoder default: 1e-4
decoder_lr = 0.05 # learning rate for decoder default: 4e-4
grad_clip = 5.  # clip gradients at an absolute value of default: 5.
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
print_freq = 100  # print training/validation stats every __ batches

class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        log_pred = torch.log(pred + 1)
        log_pred[torch.isnan(log_pred)] = 0

        log_actual = torch.log(actual + 1)
        log_actual[torch.isnan(log_actual)] = 0
        return self.mse(log_pred, log_actual)




def main():
    """
    Training and validation.
    """
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
    decoder = models.LoSDecoderWithAttention(embed_dim=emb_dim,
                                             decoder_dim=decoder_dim,
                                             vocab_size=len(ward_map),
                                             encoder_dim=encoder_dim,
                                             dropout=dropout)
    # encoder and decoder parameters
    print("\nNumber of trainable encoder parameters: %d\n" % (utils.count_parameters(encoder),))
    print("\nNumber of trainable decoder parameters: %d\n" % (utils.count_parameters(decoder),))

    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=decoder_lr)

    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    # criterion = nn.MSELoss().to(device)
    criterion = MSLELoss().to(device)

    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(
        datasets.PathwayDataset(data_folder, 'TRAIN'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.PathwayDataset(data_folder, 'VAL'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in score
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 4 == 0:
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

        # If this was the first epoch, get the best score
        if epoch == start_epoch:
            best_score = recent_score
            is_best = True
        else:
            # Check if there was an improvement
            is_best = recent_score < best_score
            best_score = min(recent_score, best_score)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

        # Save checkpoint
        utils.save_checkpoint(model_folder, model_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
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
        scores, los_sorted, decode_lengths, alphas, sort_ind = decoder(enc, pathway, p_len, los)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = los_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        scores = scores.squeeze(1)

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
        mse, rmse, mae, mape = utils.los_accuracy(scores, targets)
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Actual Loss {loss:.3f}\t'
                  'MSE Loss {mse_loss:.3f}\t'
                  'RMSE Loss {rmse_loss:.3f}\t'
                  'MAE Loss {mae_loss:.3f}\t'
                  'MAPE Loss {mape_loss:.3f}\t'.format(epoch, i, len(train_loader),
                                                       batch_time=batch_time,
                                                       data_time=data_time,
                                                       loss=loss.item(),
                                                       mse_loss=mse,
                                                       rmse_loss=rmse,
                                                       mae_loss=mae,
                                                       mape_loss=mape))


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
    losses = utils.AverageMeter()  # loss

    start = time.time()

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
            scores, los_sorted, decode_lengths, alphas, sort_ind = decoder(enc, pathway, p_len, los)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = los_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            scores = scores.squeeze(1)

            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item())
            mse, rmse, mae, mape = utils.los_accuracy(scores, targets)
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Actual Loss {loss:.3f}\t'
                      'Avg Actual Loss {avg_loss:.3f}\t'
                      'MSE Loss {mse_loss:.3f}\t'
                      'RMSE Loss {rmse_loss:.3f}\t'
                      'MAE Loss {mae_loss:.3f}\t'
                      'MAPE Loss {mape_loss:.3f}\t'.format(i, len(val_loader), batch_time=batch_time,
                                                           loss=loss.item(),
                                                           avg_loss=losses.avg,
                                                           mse_loss=mse,
                                                           rmse_loss=rmse,
                                                           mae_loss=mae,
                                                           mape_loss=mape))

    return losses.avg


if __name__ == '__main__':
    main()