import torch
import torch.nn as nn
import network
import matplotlib.pyplot as plt
import dataloader
import copy
import time
import validation

if __name__ == '__main__':
    device  = torch.device('cuda:0')
    dataroot = "./DLStudio-2.2.2/Examples/data/"
    dataset_archive_test = "sentiment_dataset_test_200.tar.gz"
    path_to_saved_embeddings = "/home/yangbj/695/hw8/word2vec/"
    batch_size = 1
    dataserver_test = dataloader.SentimentAnalysisDataset(
        train_or_test='test',
        dataroot=dataroot,
        dataset_file=dataset_archive_test,
        path_to_saved_embeddings=path_to_saved_embeddings,
    )
    test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=2)

    model = network.GRUNet(input_size=300, hidden_size=100, output_size=2, num_layers=2)

    number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_layers = len(list(model.parameters()))
    print("\n\nThe number of layers in the model: %d" % num_layers)
    print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)

    ## TESTING:
    print("\nStarting testing\n")
    validation.run_code_for_testing_text_classification_with_GRU_word2vec(path_saved_model = "./saved_GRU1e-4", test_dataloader = test_dataloader, net=model)