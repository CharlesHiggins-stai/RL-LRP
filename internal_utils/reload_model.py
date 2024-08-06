from collections import OrderedDict
import torch

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
        


def get_pretrained_model(checkpoint_path, model_skeleton):
    """Get the learner model."""
    # Load the model
    empty_model = model_skeleton()
    # Load the model weights to the checkpoint
    # now cascade down various update patches to load model
    try:
        if not torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_path)
        checkpoint = update_dictionary_patch(checkpoint)
        empty_model.load_state_dict(checkpoint['state_dict'])
        
        return empty_model
    except Exception as e:
        print(f"Something went wrong in loading the model: {e}")
        raise ValueError("Unable to load the model")