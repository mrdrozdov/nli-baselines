import torch
import os

def get_l2_loss(model, l2_lambda):
    loss = 0.0
    for w in model.parameters():
        loss += l2_lambda * torch.sum(torch.pow(w, 2))
    return loss

def ckpt_path(save_path, filename=None, best=False):
    """
    If save_path is a file, use this path.

    Otherwise, join save_path and filename.

    If file does not end with suffix or best_suffix, then append suffix.

    If best==True, then append best_suffix.
    """
    if os.path.isfile(save_path):
        return save_path
    if not filename:
        raise Exception("If save_path is not a file, must provide a filename.")
    suffix = '.ckpt'
    best_suffix = '.ckpt'
    save_path = os.path.join(save_path, filename)
    save_path = os.path.expanduser(save_path)
    if not save_path.endswith(suffix) and not save_path.endswith(best_suffix):
        save_path = save_path + suffix
    if best:
        save_path = save_path + '_best'
    return save_path

def pack_checkpoint(step, best_dev_error, model, opt):
    save_dict = dict(step=step, best_dev_error=best_dev_error,
        model=model, opt=opt)
    return save_dict

def save(save_dict, filename):
    to_save = dict()
    if 'step' in save_dict:
        to_save['step'] = save_dict['step']
    if 'best_dev_error' in save_dict:
        to_save['best_dev_error'] = save_dict['best_dev_error']
    if 'model' in save_dict:
        to_save['model_state_dict'] = save_dict['model'].state_dict()
    if 'opt' in save_dict:
        to_save['opt_state_dict'] = save_dict['opt'].state_dict()

    torch.save(to_save, filename)
