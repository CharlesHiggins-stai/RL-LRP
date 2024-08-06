# workflow for the visualisation and data analsysis

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from internal_utils import get_data_imagenette, get_teacher_model, preprocess_images, add_random_noise_batch, blur_image_batch, compute_distance_between_images, compute_sparseness_of_heatmap, imagenette_to_imagenet_label_mapping_fast
from .evaluation_functions import perform_lrp_plain, perform_gradcam
TRUNCATE = 10
# Load data
# generate the blurred images, noisy images, and the ground truth heatmap images
# then for each, calculate the distance between the heatmaps over the blurred images and the ground truth heatmap images
def process_batch(
    input_batch:torch.Tensor, 
    input_labels:torch.Tensor,  
    methods: list, 
    kernel_size_min: float, 
    kernel_size_max:float, 
    noise_level_min: float, 
    noise_level_max: float):
    """Process the batch of images.

    Args:
        model (torch.nn.Module): model to be visualised
        methods (list): list of methods(functions) to be used on each datapoint of form (name, method, model)
        kernel_size (int): size of the Gaussian kernel
        noise_level (float): level of noise to be added
    Returns:
        dict: dictionary of distances between the heatmaps
    """
    results_dictionary = {}
    for name, method, model in methods:
        # print(f"Processing method {name}")
        # get the ground truth heatmap using the method
        classifications, ground_truth_heatmap = method(input_batch, input_labels, model)
        # print('generated the ground truth heatmap')
        # treat various images to get the noisy and blurred images
        # run preprecoessing on the images --- normalise them to be within the right range
        noisy_images_small = preprocess_images(add_random_noise_batch(input_batch, noise_level_min))
        noisy_images_large = preprocess_images(add_random_noise_batch(input_batch, noise_level_max))
        blurred_images_small = preprocess_images(blur_image_batch(input_batch, kernel_size_min))
        blurred_images_large = preprocess_images(blur_image_batch(input_batch, kernel_size_max))
        # print('Blurred and noised up images')
        # generate the new heatmaps for each
        noisy_heatmaps_small_classifications, noisy_heatmaps_small = method(noisy_images_small, input_labels, model)
        noisy_heatmaps_large_classifications, noisy_heatmaps_large = method(noisy_images_large, input_labels, model)
        blurred_heatmaps_small_classifications, blurred_heatmaps_small = method(blurred_images_small, input_labels, model)
        blurred_heatmaps_large_classifications, blurred_heatmaps_large = method(blurred_images_large, input_labels, model)
        # print('Generated heatmaps')
        # Prune results so that a small noise doesn't change classification, and a large noise does change the classification
        noisy_heatmaps_small_class_change = correct_classifcation_column(noisy_heatmaps_small_classifications, classifications)
        noisy_heatmaps_large_class_change = correct_classifcation_column(noisy_heatmaps_large_classifications, classifications)
        blurred_heatmaps_small_class_change = correct_classifcation_column(blurred_heatmaps_small_classifications, classifications)
        blurred_heatmaps_large_class_change = correct_classifcation_column(blurred_heatmaps_large_classifications, classifications)
        # print(f"Pruned batch")
        # calculate the distance between the heatmaps
        distance_noise_small = compute_distance_between_images(ground_truth_heatmap, noisy_heatmaps_small)
        distance_noise_large = compute_distance_between_images(ground_truth_heatmap, noisy_heatmaps_large)
        distance_blur_small = compute_distance_between_images(ground_truth_heatmap, blurred_heatmaps_small)
        distance_blur_large = compute_distance_between_images(ground_truth_heatmap, blurred_heatmaps_large)
        # calculate sparseness of heatmap
        sparseness_original, sparseness_gini = compute_sparseness_of_heatmap(ground_truth_heatmap)
        # store the results in the dictionary
        results_dictionary[f"{name}_distance_noise_small"] = distance_noise_small
        results_dictionary[f"{name}_distance_noise_small_class_change"] = noisy_heatmaps_small_class_change
        results_dictionary[f"{name}_distance_noise_large"] = distance_noise_large
        results_dictionary[f"{name}_distance_noise_large_class_change"] = noisy_heatmaps_large_class_change
        results_dictionary[f"{name}_distance_blur_small"] = distance_blur_small
        results_dictionary[f"{name}_distance_blur_small_class_change"] = blurred_heatmaps_small_class_change
        results_dictionary[f"{name}_distance_blur_large"] = distance_blur_large
        results_dictionary[f"{name}_distance_blur_large_class_change"] = blurred_heatmaps_large_class_change
        results_dictionary[f"{name}_sparseness_original"] = sparseness_original
        results_dictionary[f"{name}_sparseness_gini"] = sparseness_gini
    # return data
    return results_dictionary


def correct_classifcation_column(input_tensor: torch.Tensor, labels: torch.Tensor):
    """Return a tensor with boolean value indicating if the classification was correct.

    Args:
        input_tensor (torch.Tensor): tensor to be pruned
        labels (torch.Tensor): labels of the tensor
    Returns:
        torch.Tensor: pruned tensor
    """
    classification_changed_tensor = input_tensor == labels
    assert classification_changed_tensor.shape[0] == input_tensor.shape[0], "The tensor should have the same number of elements"
    return classification_changed_tensor


def prune_tensor_to_only_correct_indicies(input_tensor: torch.Tensor, 
                                          input_tensor_heatmaps:torch.Tensor, 
                                          labels: torch.Tensor, 
                                          ground_truth_heatmaps:torch.Tensor, 
                                          inverse:bool = False):
    """Prune the tensor to only include the correct indicies.

    Args:
        input_tensor (torch.Tensor): tensor to be pruned
        input_tensor_heatmaps (torch.Tensor): tensor of heatmaps to be pruned
        
        labels (torch.Tensor): labels of the tensor
        ground_truth_heatmaps (torch.Tensor): ground truth heatmaps
    Returns:
        torch.Tensor: pruned tensor
    """
    # prune the tensor
    if not inverse:
        correct_indicies = torch.where(input_tensor == labels)
        pruned_tensor_output = input_tensor[correct_indicies]
        pruned_tensor_heatmaps = input_tensor_heatmaps[correct_indicies]
        pruned_ground_truth_heatmaps = ground_truth_heatmaps[correct_indicies]
    else: 
        incorrect_indicies = torch.where(input_tensor != labels)
        pruned_tensor = input_tensor[incorrect_indicies]
        pruned_tensor_heatmaps = input_tensor_heatmaps[incorrect_indicies]
        pruned_ground_truth_heatmaps = ground_truth_heatmaps[incorrect_indicies]
        
    return pruned_tensor_heatmaps, pruned_ground_truth_heatmaps


def evaluate_performance(model, train_data, test_data, convert_to_imagenet_labels = True):
    """Evaluate the performance of the model on the data"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loss, train_accuracy = [], []
    test_loss, test_accuracy = [], []
    model.to(device)
    for i, (input, label) in tqdm(enumerate(train_data)):
    # for i in tqdm(range(0, TRUNCATE)):
        # input, label = next(iter(train_data))
        input = preprocess_images(input).to(device)
        if convert_to_imagenet_labels:
            label = imagenette_to_imagenet_label_mapping_fast(label).to(device)
        else:
            label = label.to(device)
        with torch.no_grad():
            output = model(input)
        loss = torch.nn.functional.cross_entropy(output, label)
        accuracy = (output.argmax(dim=1) == label).float().mean() * 100
        train_loss.append(loss)
        train_accuracy.append(accuracy)
    for i, (input, label) in tqdm(enumerate(test_data)):
    # for i in tqdm(range(0, TRUNCATE)):
        # input, label = next(iter(test_data))
        input = preprocess_images(input).to(device)
        if convert_to_imagenet_labels:
            label = imagenette_to_imagenet_label_mapping_fast(label).to(device)
        else:
            label = label.to(device)
        with torch.no_grad():
            output = model(input)
        loss = torch.nn.functional.cross_entropy(output, label)
        accuracy = (output.argmax(dim=1) == label).float().mean() * 100
        test_loss.append(loss)
        test_accuracy.append(accuracy)
    # convert results to dictionary
    results_train = {
        "train_loss": np.array(train_loss),
        "train_accuracy": np.array(train_accuracy)
    }
    results_test = {
        "test_loss": np.array(test_loss),
        "test_accuracy": np.array(test_accuracy)
    }
    return pd.DataFrame.from_dict(results_train), pd.DataFrame.from_dict(results_test)   

def process_dataset(data_loader, methods, kernel_size_min, kernel_size_max, noise_level_min, noise_level_max, convert_to_imagenet_labels = True):
    """Generate results over a single dataset

    Args:
        data_loader (DataLoader object): Dataloader object without heavy preprocessing steps
        methods (List of Tuples): list of methods(functions) to be used on each datapoint of form (name, method, model)
        kernel_size_min (int): For Gaussian Blur, minimum kernel size
        kernel_size_max (int): For Gaussian Blur, maximum kernel size
        noise_level_min (float): For adding noise, minimum noise level
        noise_level_max (float): For adding noise, maximum noise level

    Returns:
        dict: dictionary of results with keys as method names and values as torch tensors of results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = {}
    for i, (input_batch, input_labels) in tqdm(enumerate(data_loader)):
    # for i in tqdm(range(0, TRUNCATE)):
        # input_batch, input_labels = next(iter(data_loader))
        input_batch.to(device)
        if convert_to_imagenet_labels:
            input_labels = imagenette_to_imagenet_label_mapping_fast(input_labels).to(device)  
        else:
            input_labels = input_labels.to(device)  
        results = process_batch(
            input_batch, 
            input_labels, 
            methods, 
            kernel_size_min, 
            kernel_size_max, 
            noise_level_min, 
            noise_level_max
        )
        for key, value in results.items():
            # deal with empty tensors
            if value == None:
                    value = torch.empty(0)
            # add to table
            if key not in table.keys():
                table[key] = value.detach()
            else:
                table[key] = torch.cat([table[key], value.detach()], dim = 0)
        # print(f"Processed batch {i+1}/{len(data_loader)}")
    return table    

def evaluate_explanations(train_data, test_data, methods, save_results = False, convert_to_imagenet_labels = True):
    """Evaluate the explanations of the model on the data
    
    N.B. Methods is a list in the form (name, method, model) where name is a string, method is a function to generate a heatmap on a certain model
    
    """
    
    # define params
    kernel_size_min = 3
    kernel_size_max = 5
    noise_level_min = 0.1
    noise_level_max = 0.2
    
    # process datasets
    train_table = process_dataset(train_data, methods, kernel_size_min, kernel_size_max, noise_level_min, noise_level_max, convert_to_imagenet_labels)
    test_table = process_dataset(test_data, methods, kernel_size_min, kernel_size_max, noise_level_min, noise_level_max, convert_to_imagenet_labels)
    # convert to pandas dataframe
    df_train = pd.DataFrame(train_table)
    df_test = pd.DataFrame(test_table)
    
    # save results
    if save_results:
        df_train.to_csv("train_results.csv")
        df_test.to_csv("test_results.csv")
    return df_train, df_test



def main():
    # define params
    kernel_size_min = 3
    kernel_size_max = 5
    noise_level_min = 0.1
    noise_level_max = 0.2
    # get the data
    data_loader = get_data_imagenette()
    # get the model
    # learner_model = get_learner_model()
    teacher_model = get_teacher_model()
    # define the methods
    methods = [
        # ("LRP", perform_lrp_plain, WrapperNet(teacher_model, hybrid_loss=True)),
        # ("LossLRP", perform_loss_lrp, learner_model),
        ("GradCAM", perform_gradcam, teacher_model),
    ]
    # process the data
    table = {}
    # for i, (input_batch, input_labels) in enumerate(data_loader):
    for _ in range(0,3):
        input_batch, input_labels = next(iter(data_loader))
        results = process_batch(
            input_batch, 
            input_labels, 
            methods, 
            kernel_size_min, 
            kernel_size_max, 
            noise_level_min, 
            noise_level_max
        )
        # print the results
        # print(f"Batch {i} results: {results}")
        for key, value in results.items():
            if key not in table.keys():
                table[key] = value.detach()
            else:
                table[key] = torch.cat([table[key], value.detach()], dim = 0)
    # convert to pandas dataframe
    df = pd.DataFrame(table)
    # save results
    df.to_csv("test_results.csv")
    