import torch
import torch.nn as nn
import network
import matplotlib.pyplot as plt
import dataloader
import copy
import time
import validation


def run_code_for_training_for_text_classification_with_GRU_word2vec(epochs, device, learning_rate, momentum, train_dataloader, path_saved_model, net, display_train_loss=False):
    filename_for_out = "performance_numbers_" + str(epochs) + ".txt"
    FILE = open(filename_for_out, 'w')
    net = copy.deepcopy(net)
    net = net.to(device)
    ##  Note that the GREnet now produces the LogSoftmax output:
    criterion = nn.NLLLoss()
    accum_times = []
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    training_loss_tally = []
    start_time = time.perf_counter()
    for epoch in range(epochs):
        print("")
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            review_tensor, category, sentiment = data['review'], data['category'], data['sentiment']
            review_tensor = review_tensor.to(device)
            sentiment = sentiment.to(device)
            ## The following type conversion needed for MSELoss:
            ##sentiment = sentiment.float()
            optimizer.zero_grad()
            hidden = net.init_hidden().to(device)
            for k in range(review_tensor.shape[1]):
                output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0, k], 0), 0), hidden)
            loss = criterion(output, torch.argmax(sentiment, 1))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if i % 200 == 199:
                avg_loss = running_loss / float(200)
                training_loss_tally.append(avg_loss)
                current_time = time.perf_counter()
                time_elapsed = current_time - start_time
                print("[epoch:%d  iter:%4d  elapsed_time:%4d secs]     loss: %.5f" % (
                epoch + 1, i + 1, time_elapsed, avg_loss))
                accum_times.append(current_time - start_time)
                FILE.write("%.5f\n" % avg_loss)
                FILE.flush()
                running_loss = 0.0
    torch.save(net.state_dict(), path_saved_model)
    print("Total Training Time: {}".format(str(sum(accum_times))))
    print("\nFinished Training\n\n")
    if display_train_loss:
        plt.figure(figsize=(10, 5))
        plt.title("GRU Training Loss vs. Iterations")
        plt.plot(training_loss_tally)
        plt.xlabel("iterations")
        plt.ylabel("training loss")
        plt.legend()
        plt.savefig("training_loss_GRU1e-4.png")
        plt.show()

if __name__ == '__main__':
    device  = torch.device('cuda:0')
    dataroot = "./DLStudio-2.2.2/Examples/data/"
    dataset_archive_train = "sentiment_dataset_train_200.tar.gz"
    dataset_archive_test = "sentiment_dataset_test_200.tar.gz"
    path_to_saved_embeddings = "/home/yangbj/695/hw8/word2vec/"
    batch_size = 1
    dataserver_train = dataloader.SentimentAnalysisDataset(
        train_or_test='train',
        dataroot=dataroot,
        dataset_file=dataset_archive_train,
        path_to_saved_embeddings=path_to_saved_embeddings,
    )
    dataserver_test = dataloader.SentimentAnalysisDataset(
        train_or_test='test',
        dataroot=dataroot,
        dataset_file=dataset_archive_test,
        path_to_saved_embeddings=path_to_saved_embeddings,
    )
    train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                                                   batch_size=batch_size, shuffle=True,
                                                   num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=2)

    model = network.GRUNet(input_size=300, hidden_size=100, output_size=2, num_layers=2)

    number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_layers = len(list(model.parameters()))
    print("\n\nThe number of layers in the model: %d" % num_layers)
    print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)

    ## TRAINING:
    print("\nStarting training\n")
    run_code_for_training_for_text_classification_with_GRU_word2vec(epochs=5, device=device, learning_rate =  1e-4,
                                                                    momentum = 0.9, train_dataloader=train_dataloader,
                                                                    path_saved_model = "./saved_GRU1e-4", net=model, display_train_loss=True)

    ## TESTING:
    print("\nStarting testing\n")
    validation.run_code_for_testing_text_classification_with_GRU_word2vec(path_saved_model = "./saved_GRU1e-4", test_dataloader = test_dataloader, net=model)