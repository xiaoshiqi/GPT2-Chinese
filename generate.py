# -*- encoding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
from operator import add
import argparse
from tqdm import trange
from transformers import GPT2LMHeadModel
import numpy as np
from tokenizations import tokenization_bert

SMALL_CONST = 1e-15

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    #assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def to_var(x, requires_grad=False, volatile=False, device="cuda"):
    if torch.cuda.is_available() and device == "cuda":
        x = x.cuda()
    elif device != "cuda":
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

    
# quote from https://github.com/uber-research/PPLM
def perturb_past(
    past,
    model,
    prev,
    unpert_past=None,
    unpert_logits=None,
    accumulated_hidden=None,
    grad_norms=None,
    stepsize=0.01,
    one_hot_bows_vectors=None,
    num_iterations=3,
    horizon_length=1,
    window_length=0,
    gamma=1.5,
    kl_scale=0.01,
    device="cuda",
):

    # Generate inital perturbed past
    grad_accumulator = [(np.zeros(p.shape).astype("float32")) for p in past]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    # TODO fix this comment (SUMANTH)
    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, curr_length, _ = past[0].shape

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = tuple(past[0].shape[:-2]) + tuple([window_length]) + tuple(past[0].shape[-1:])

        zeros_key_val_shape = (
            tuple(past[0].shape[:-2]) + tuple([curr_length - window_length]) + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat((ones_mask, torch.zeros(zeros_key_val_shape)), dim=-2).to(device)
    else:
        window_mask = torch.ones_like(past[0]).to(device)

        

    
    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    for i in range(num_iterations):
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device) for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation))
        _, _, _, curr_length, _ = curr_perturbation[0].shape
        all_logits, _, all_hidden = model(prev, past=perturbed_past)
        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1).detach()
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []

        for one_hot_bow in one_hot_bows_vectors:
            bow_logits = torch.mm(probs, torch.t(one_hot_bow))
            bow_loss = -torch.log(torch.sum(bow_logits))
            loss += bow_loss
            loss_list.append(bow_loss)



        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = unpert_probs + SMALL_CONST * (unpert_probs <= SMALL_CONST).float().to(device).detach()
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(device).detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * ((corrected_probs * (corrected_probs / unpert_probs).log()).sum())
            loss += kl_loss

        loss.backward(retain_graph=True)

        # calculate gradient norms
        if grad_norms is not None:
            grad_norms = [
                torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
                for index, p_ in enumerate(curr_perturbation)
            ]
        else:
            grad_norms = [
                (torch.norm(p_.grad * window_mask) + SMALL_CONST) for index, p_ in enumerate(curr_perturbation)
            ]

        # normalize gradients
        grad = [
            -stepsize * (p_.grad * window_mask / grad_norms[index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [to_var(torch.from_numpy(p_), requires_grad=True, device=device) for p_ in grad_accumulator]
    pert_past = list(map(add, past, grad_accumulator))

    return pert_past, new_accumulated_hidden, grad_norms
    
    
def sample_sequence(model, context, length, temperature=1.0, top_k=30, top_p=0.0, device='cpu', bow=False, bow_vectors=None, bow_stepsize=0, bow_num_iterations=3,tokenizer=None):
    inputs = torch.LongTensor(context).view(1, -1).to(device)
    
    generate = [] + context
    grad_norms = None
    
    _, past, _ = model(inputs[:, :-1])
    prev = inputs[:, -1].view(1, -1)

    for i in trange(length):
        unpert_logits, unpert_past, unpert_all_hidden = model(inputs)
        unpert_last_hidden = unpert_all_hidden[-1]

        if not bow:
            pert_past = past
        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            pert_past, _, grad_norms = perturb_past(
                past,
                model,
                prev,
                unpert_past=unpert_past,
                unpert_logits=unpert_logits,
                accumulated_hidden=accumulated_hidden,
                grad_norms=grad_norms,
                stepsize=bow_stepsize,
                one_hot_bows_vectors=bow_vectors,
                num_iterations=bow_num_iterations,
                window_length=0,                    #no masking
                gamma=1.5,
                kl_scale=0.01,                      ########################
                device=device,
            )

        output, past, _ = model(prev, past=pert_past)
        output = output[-1].squeeze(0) / temperature
        
        gm_scale = 0.99                             ##########################
        if bow:
            unpert_probs = torch.softmax(unpert_logits[:, -1, :], dim=-1)
            pert_probs = torch.softmax(output, dim=-1)
            pert_probs = (pert_probs ** gm_scale) * (unpert_probs ** (1 - gm_scale))  # + SMALL_CONST
            
            filtered_logits = top_k_top_p_filtering(pert_probs, top_k=top_k, top_p=top_p, filter_value=0)
            filtered_logits = filtered_logits / torch.sum(filtered_logits)
        
        else:
            filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
            filtered_logits = torch.softmax(filtered_logits, dim=-1)
        
        prev = torch.multinomial(filtered_logits, num_samples=1)

        if tokenizer.convert_ids_to_tokens(prev.item()) == "[PAD]" or tokenizer.convert_ids_to_tokens(prev.item()) == "[CLS]":
            break

        generate.append(prev.item())
    return generate
    
def build_bow_vectors(pathes, tokenizer, device='cpu'):
    pathes = pathes.split(";")
    vectors = []
    for path in pathes:
        with open(path, "r", encoding='UTF-8') as f:
            words = f.read().strip().split("\n")
        bag = [tokenizer.encode(word.strip()) for word in words]
        
        bag = list(filter(lambda x: len(x) <= 1, bag))
        bag = torch.tensor(bag).to(device)
        vector = torch.zeros(bag.shape[0], tokenizer.vocab_size).to(device)
        vector.scatter_(1, bag, 1)
        vectors.append(vector)
        
    return vectors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='cuda visible devices')
    parser.add_argument('--nsamples', default=8, type=int, required=False, help='number of generated samples')
    parser.add_argument('--temperature', default=1, type=float, required=False)
    parser.add_argument('--topk', default=8, type=int, required=False, help='k for top k sampling')
    parser.add_argument('--topp', default=0, type=float, required=False, help='p for top p sampling')
    parser.add_argument('--tokenizer_path', default='data/vocabs.txt', type=str, required=False, help='path of the vocabulary file')
    parser.add_argument('--model_path', default='model/model_epoch24', type=str, required=False, help='pre-trained model dir')
    parser.add_argument('--prefix', default='仁义礼智信', type=str, required=False, help='prefix of the couplet')
    parser.add_argument('--save_samples', action='store_true', help='save samples')
    parser.add_argument('--save_samples_path', default='data', type=str, required=False, help="save the samples to this dir")

    parser.add_argument('--bow', action='store_true', help='use PPLM-BOW')
    parser.add_argument('--bow_path', default='data/bow_newyear.txt', type=str, required=False, help='path of the bag of considered characters')
    parser.add_argument('--bow_stepsize', default=0.3, type=float, required=False, help='stepsize of the PPLM')
    parser.add_argument('--bow_num_iterations', default=3, type=int, required=False, help='num_iterations of the PPLM')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = "cuda" if torch.cuda.is_available() else "cpu"


    tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path, output_hidden_states=True)
    model.to(device)
    model.eval()
    
    if args.prefix.find("|")<0:
        length = model.config.n_ctx - len(args.prefix)
    else:
        length = 2*args.prefix.index("|") - len(args.prefix) + 1

    bow_vectors = None
    if args.bow:
        bow_vectors = build_bow_vectors(args.bow_path, tokenizer, device)

    if args.save_samples:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/generated.txt', 'w', encoding='utf8')
        
    context_tokens = [tokenizer.convert_tokens_to_ids('[MASK]')]+tokenizer.convert_tokens_to_ids(tokenizer.tokenize(args.prefix))
    
    for t in range(args.nsamples):
        out = sample_sequence(model, context_tokens, length, temperature=args.temperature, top_k=args.topk, top_p=args.topp, device=device, bow=args.bow, bow_vectors=bow_vectors, bow_stepsize=args.bow_stepsize, bow_num_iterations=args.bow_num_iterations, tokenizer=tokenizer)
        text = tokenizer.convert_ids_to_tokens(out)
        for i, item in enumerate(text):
            if item == '[MASK]':
                text[i] = '上联：'
            elif item == '[PAD]':
                text[i] = ' '   
            elif item == '|':
                text[i] = ' 下联：'                      

        print("=" * 40 + " SAMPLE " + str(t+1) + " " + "=" * 40 + "\n")
        text = ''.join(text)
        print(text)
        if args.save_samples:
            samples_file.write(text+'\n')

    if args.save_samples:
        samples_file.close()

        
if __name__ == '__main__':
    main()
