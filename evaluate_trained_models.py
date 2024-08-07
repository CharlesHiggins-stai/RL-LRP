from internal_utils import get_CIFAR_10_dataloader_without_normalization, get_pretrained_model
from experiments import WrapperNet, perform_lrp_plain, evaluate_explanations
from baselines.trainVggBaselineForCIFAR10 import vgg


if __name__ == "__main__":

    # Load the models
    hybrid_model = get_pretrained_model("/Users/charleshiggins/Personal/CharlesPhD/CodeRepo/xai_intervention/RL-LRP/model_files/checkpoint_299_2024-08-06_02-23-55_default.tar", vgg.vgg11)
    baseline_model  = get_pretrained_model("/Users/charleshiggins/Personal/CharlesPhD/CodeRepo/xai_intervention/RL-LRP/model_files/checkpoint_299_2024-08-06_11-16-09_sanity_check.tar", vgg.vgg11)

    # Load test and train data to run evaluation over
    train_data, test_data = get_CIFAR_10_dataloader_without_normalization(train=True, batch_size=8, num_workers=4, pin_memory=True), get_CIFAR_10_dataloader_without_normalization(train=False, batch_size=8, num_workers=4, pin_memory=True)
    methods = [
        ("HYBRID", perform_lrp_plain, WrapperNet(hybrid_model, hybrid_loss=True)),
        ("BASELINE", perform_lrp_plain, WrapperNet(baseline_model, hybrid_loss=True))
    ]
    # now evaluate explanations
    print("WARNING: EVAULUATING EXPLANATIONS IS TOO COMPUTATIONALLY INTENSE FOR A NOTEBOOK. WE SUGGEST RUNNING THIS IN A SCRIPT, AND LOADING THE RESULTS FROM .csv FILES FOR ANALYSIS.")
    df_train, df_test = evaluate_explanations(train_data, test_data, methods, save_results = True, convert_to_imagenet_labels=False)
    print('run complete')
