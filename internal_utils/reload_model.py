
from collections import OrderedDict


def update_dictionary_patch(checkpoint):
    """
    Saving the checkpoint fails to inlcude the correct naming --- at times the model is saved with 'module' prefix, when it shouldn't be. 
    This is a quick patch to avoid having to re-write the logic for building the layers in the internals of the model 
    and be able to use the model as is. 
    ##################################################################################################
    Assumes that the checkpoint has already been loaded into memory via torch.load(path_to_checkpoint)
    ##################################################################################################
    """
    new_state_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if 'module' in key:
            new_key = key.replace('module.', '')
        else:
            new_key = key
        new_state_dict[new_key]= value
    checkpoint['new_state_dict'] = new_state_dict
    checkpoint['state_dict'] = new_state_dict
    checkpoint = update_dictionary_patch_3(update_dictionary_patch_2(checkpoint))
    return checkpoint


def update_dictionary_patch_2(checkpoint):
    new_state_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if "_orig_mod." in key:
            new_key = key.replace("_orig_mod.", "")
        else:
            new_key = key
        new_state_dict[new_key] = value
    checkpoint["new_state_dict"] = new_state_dict
    checkpoint['state_dict'] = new_state_dict
    return checkpoint


def update_dictionary_patch_3(checkpoint):
    new_state_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if "model." in key:
            new_key = key.replace("model.", "")
        else:
            new_key = key
        new_state_dict[new_key] = value
    checkpoint["new_state_dict"] = new_state_dict
    checkpoint['new_state_dict'] = new_state_dict
    checkpoint['state_dict'] = new_state_dict
    return checkpoint
        