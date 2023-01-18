import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn

def run_code_for_testing_text_classification_with_GRU_word2vec(path_saved_model, test_dataloader, net):
    net.load_state_dict(torch.load(path_saved_model))
    classification_accuracy = 0.0
    negative_total = 0
    positive_total = 0
    confusion_matrix = torch.zeros(2, 2)
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            review_tensor, category, sentiment = data['review'], data['category'], data['sentiment']
            hidden = net.init_hidden()
            for k in range(review_tensor.shape[1]):
                output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0, k], 0), 0), hidden)
            predicted_idx = torch.argmax(output).item()
            gt_idx = torch.argmax(sentiment).item()
            if i % 100 == 99:
                print("   [i=%d]    predicted_label=%d       gt_label=%d" % (i + 1, predicted_idx, gt_idx))
            if predicted_idx == gt_idx:
                classification_accuracy += 1
            if gt_idx == 0:
                negative_total += 1
            elif gt_idx == 1:
                positive_total += 1
            confusion_matrix[gt_idx, predicted_idx] += 1

    # plot confusion matrix
    plt.figure(figsize=(10, 7))
    seaborn.heatmap(confusion_matrix, annot=True, linewidths=.5,
                    xticklabels=['predicted negative', 'predicted positive'], yticklabels=['true negative', 'true positive'])
    plt.title("Net" + 'Accuracy:' + str(float(classification_accuracy) * 100 / float(i)) + '%')
    plt.savefig("net_confusion_matrix_GRU.png")

    print("\nOverall classification accuracy: %0.2f%%" % (float(classification_accuracy) * 100 / float(i)))
    out_percent = np.zeros((2, 2), dtype='float')
    out_percent[0, 0] = "%.3f" % (100 * confusion_matrix[0, 0] / float(negative_total))
    out_percent[0, 1] = "%.3f" % (100 * confusion_matrix[0, 1] / float(negative_total))
    out_percent[1, 0] = "%.3f" % (100 * confusion_matrix[1, 0] / float(positive_total))
    out_percent[1, 1] = "%.3f" % (100 * confusion_matrix[1, 1] / float(positive_total))
    print("\n\nNumber of positive reviews tested: %d" % positive_total)
    print("\n\nNumber of negative reviews tested: %d" % negative_total)
    print("\n\nDisplaying the confusion matrix:\n")
    out_str = "                      "
    out_str += "%18s    %18s" % ('predicted negative', 'predicted positive')
    print(out_str + "\n")
    for i, label in enumerate(['true negative', 'true positive']):
        out_str = "%12s:  " % label
        for j in range(2):
            out_str += "%18s%%" % out_percent[i, j]
        print(out_str)