import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from pathway_los_prediction import models, datasets, utils, model_configuration


model_path = '..\\pathway_los_prediction_models\\cap_6_pathways_20_min_ward_freq_best_checkpoint.pth.tar'
ward_map_path = '..\\sample_datasets\\ward_map.json'
data_folder = '..\\sample_datasets'  # folder with data files saved by create_input_dataset.py
batch_size = 1 # process 1 batch at a time
workers = 1  # for data-loading;
beam_size = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pathway_vector_beam_search(encoder, decoder, ward_map, beam_size=3):
    """
    Reads an image and captions it with beam search
    :param cat_features [# cat features]
    :param cont_features [# cont features]
    :param encoder: encoder model
    :param decoder: decoder model
    :param ward_map: ward map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    vocab_size = len(ward_map)

    # Read test dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.PathwayDataset(data_folder, 'TEST'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    i_seqs = []
    i_alphas = []
    pred_df = pd.DataFrame()

    for i, (cn_features, ct_features, pathway, _, _) in enumerate(test_loader):

        # Move to GPU, if available
        cn_features = cn_features.to(device)
        ct_features = ct_features.to(device)

        # Set k
        k = beam_size

        # Encode
        encoder_out = encoder(cn_features, ct_features)  # (1, # of embeddings + # of cont features)
        encoder_dim = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, encoder_dim)  # (k, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[ward_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Tensor to store top k sequences' alphas; now they're just 1s
        seqs_alpha = torch.ones(k, 1, encoder_dim).to(device)  # (k, 1, encoder_dim)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_alpha = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, encoder_dim)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_wards = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_wards = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_ward_inds = top_k_wards / vocab_size  # (s)
            next_ward_inds = top_k_wards % vocab_size  # (s)

            # Add new words to sequences, alphas
            seqs = torch.cat([seqs[prev_ward_inds], next_ward_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            seqs_alpha = torch.cat([seqs_alpha[prev_ward_inds], alpha[prev_ward_inds].unsqueeze(1)],
                                   dim=1)  # (s, step+1, enc_dim)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_ward_inds) if
                               next_word != ward_map['<end>']]
            complete_inds = list(set(range(len(next_ward_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            h = h[prev_ward_inds[incomplete_inds]]
            c = c[prev_ward_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_ward_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_ward_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 10:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]

        # save predictions in a dictionary
        pred_dict = {'continuous_features': cn_features.cpu().detach().squeeze().unsqueeze(0).numpy().tolist(), 'categorical_features':ct_features.cpu().detach().squeeze().unsqueeze(0).numpy().tolist(), 'pathway': pathway.cpu().detach().squeeze().unsqueeze(0).numpy().tolist(),
                     'predicted_pathway': [seq]}

        data_df = pd.DataFrame(pred_dict, columns=['continuous_features', 'categorical_features', 'pathway', 'predicted_pathway'])

        # concat with results_df
        pred_df = pd.concat([pred_df, data_df], axis=0)
        i_seqs.append(seq)
        i_alphas.append(alphas)

    return i_seqs, i_alphas, pred_df


# def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
#     """
#     Visualizes caption with weights at every word.
#
#     Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
#
#     :param image_path: path to image that has been captioned
#     :param seq: caption
#     :param alphas: weights
#     :param rev_word_map: reverse word mapping, i.e. ix2word
#     :param smooth: smooth weights?
#     """
#     image = Image.open(image_path)
#     image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)
#
#     words = [rev_word_map[ind] for ind in seq]
#
#     for t in range(len(words)):
#         if t > 50:
#             break
#         plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)
#
#         plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
#         plt.imshow(image)
#         current_alpha = alphas[t, :]
#         if smooth:
#             alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
#         else:
#             alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
#         if t == 0:
#             plt.imshow(alpha, alpha=0)
#         else:
#             plt.imshow(alpha, alpha=0.8)
#         plt.set_cmap(cm.Greys_r)
#         plt.axis('off')
#     plt.show()


if __name__ == '__main__':

    # Load model
    checkpoint = torch.load(model_path, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load ward map (word2ix)
    with open(ward_map_path, 'r') as j:
        ward_map = json.load(j)
    rev_ward_map = {v: k for k, v in ward_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq, alphas, pred_df = pathway_vector_beam_search(encoder, decoder, ward_map, beam_size)
    alphas = torch.FloatTensor(alphas)

    # Visualize caption and attention of best sequence
    # visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)
