import torch, torchvision, PIL
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

def main():

    ##Task 2
    img1 = PIL.Image.open("1.jpg")
    img2 = PIL.Image.open("2.jpg") #3000*3000
    transform2 = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
    )
    img1 = transform2(img1)
    img2 = transform2(img2)
    img1 = img1[:, 1000:2000, 1000:2000]
    img2 = img2[:, 1000:2000, 1000:2000]
    print(img1.shape)

    #Calculating Histogram
    histogram1_0 = torch.histc(img1[0, :, :], bins=256, min = 0.0, max = 1.0)
    histogram1_1 = torch.histc(img1[1, :, :], bins=256, min = 0.0, max = 1.0)
    histogram1_2 = torch.histc(img1[2, :, :], bins=256, min = 0.0, max = 1.0)
    histogram2_0 = torch.histc(img2[0, :, :], bins=256, min = 0.0, max = 1.0)
    histogram2_1 = torch.histc(img2[1, :, :], bins=256, min = 0.0, max = 1.0)
    histogram2_2 = torch.histc(img2[2, :, :], bins=256, min = 0.0, max = 1.0)
    #Histogram Normalization
    histogram1_0 = histogram1_0.div(histogram1_0.sum())
    histogram1_1 = histogram1_1.div(histogram1_1.sum())
    histogram1_2 = histogram1_2.div(histogram1_2.sum())
    histogram2_0 = histogram2_0.div(histogram2_0.sum())
    histogram2_1 = histogram2_1.div(histogram2_1.sum())
    histogram2_2 = histogram2_2.div(histogram2_2.sum())

    ##Task 3
    ####################################
    # plot 1:
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].plot(histogram1_0)
    axarr[0, 1].plot(histogram1_1)
    axarr[1, 0].plot(histogram1_2)
    axarr[1, 1].imshow(img1.permute(2, 1, 0)*0.5+0.5)

    plt.suptitle("Histogram1")
    plt.savefig("Histogram1")
    ####################################
    # plot 2:
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].plot(histogram2_0)
    axarr[0, 1].plot(histogram2_1)
    axarr[1, 0].plot(histogram2_2)
    axarr[1, 1].imshow(img2.permute(2, 1, 0)*0.5+0.5)

    plt.suptitle("Histogram2")
    plt.savefig("Histogram2")
    ####################################

    ##Task 4
    distance_R = wasserstein_distance(histogram1_0, histogram2_0)
    distance_G = wasserstein_distance(histogram1_1, histogram2_1)
    distance_B = wasserstein_distance(histogram1_2, histogram2_2)
    print("Wasserstein Distance R Channel Histogram: ", distance_R)
    print("Wasserstein Distance G Channel Histogram: ", distance_G)
    print("Wasserstein Distance B Channel Histogram: ", distance_B)

    ##Task 5
    # affine = torchvision.transforms.RandomAffine(degrees = 45)
    affine_img1 = torchvision.transforms.functional.affine(
        img = img1*0.5+0.5, angle = -90, translate = (0, 0), scale = 1, shear = 0)
    histogram1_0_x = torch.histc(affine_img1[0, :, :], bins=256, min = 0.0, max = 1.0)
    histogram1_1_x = torch.histc(affine_img1[1, :, :], bins=256, min = 0.0, max = 1.0)
    histogram1_2_x = torch.histc(affine_img1[2, :, :], bins=256, min = 0.0, max = 1.0)
    histogram1_0_x = histogram1_0_x.div(histogram1_0_x.sum())
    histogram1_1_x = histogram1_1_x.div(histogram1_1_x.sum())
    histogram1_2_x = histogram1_2_x.div(histogram1_2_x.sum())
    # print("affine: ", histogram1_0_x)
    # histogram1_0_x = histogram1_0_x[1:] * (affine_img1.shape[-1] *
    #                 affine_img1.shape[-2] / (affine_img1.shape[-1] *
    #                 affine_img1.shape[-2] - histogram1_0_x[0]))
    # histogram1_1_x = histogram1_1_x[1:] * (affine_img1.shape[-1] *
    #                 affine_img1.shape[-2] / (affine_img1.shape[-1] *
    #                 affine_img1.shape[-2] - histogram1_1_x[0]))
    # histogram1_2_x = histogram1_2_x[1:] * (affine_img1.shape[-1] *
    #                 affine_img1.shape[-2] / (affine_img1.shape[-1] *
    #                 affine_img1.shape[-2] - histogram1_2_x[0]))
    ####################################
    # plot 1:
    # print("affine: ", histogram1_0_x)
    # print("original: ", histogram1_0)
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].plot(histogram1_0_x)
    axarr[0, 1].plot(histogram1_1_x)
    axarr[1, 0].plot(histogram1_2_x)
    axarr[1, 1].imshow(affine_img1.permute(2, 1, 0))

    plt.suptitle("Histogram3")
    plt.savefig("Histogram3")
    ####################################
    # plot 2:
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].plot(histogram2_0[1:])
    axarr[0, 1].plot(histogram2_1[1:])
    axarr[1, 0].plot(histogram2_2[1:])
    axarr[1, 1].imshow(img2.permute(2, 1, 0)*0.5+0.5)

    plt.suptitle("Histogram4")
    plt.savefig("Histogram4")
    ####################################
    distance_R_x = wasserstein_distance(histogram1_0_x, histogram2_0[1:])
    distance_G_x = wasserstein_distance(histogram1_1_x, histogram2_1[1:])
    distance_B_x = wasserstein_distance(histogram1_2_x, histogram2_2[1:])
    print("After Affine Transformation, Wasserstein Distance R Channel Histogram: ",
          distance_R_x)
    print("After Affine Transformation, Wasserstein Distance G Channel Histogram: ",
          distance_G_x)
    print("After Affine Transformation, Wasserstein Distance B Channel Histogram: ",
          distance_B_x)

    ##Task 6
    transformer_per = torchvision.transforms.RandomPerspective(
        distortion_scale=0.6, p=1.0)
    perspective_img1 = transformer_per(img1*0.5+0.5)
    histogram1_0_x = torch.histc(perspective_img1[0, :, :], bins=256, min=0.0, max=1.0)
    histogram1_1_x = torch.histc(perspective_img1[1, :, :], bins=256, min=0.0, max=1.0)
    histogram1_2_x = torch.histc(perspective_img1[2, :, :], bins=256, min=0.0, max=1.0)
    histogram1_0_x = histogram1_0_x.div(histogram1_0_x.sum())
    histogram1_1_x = histogram1_1_x.div(histogram1_1_x.sum())
    histogram1_2_x = histogram1_2_x.div(histogram1_2_x.sum())
    ####################################
    histogram1_0_x = histogram1_0_x[1:] * (perspective_img1.shape[-1]*
                     perspective_img1.shape[-2]/(perspective_img1.shape[-1]*
                      perspective_img1.shape[-2]-histogram1_0_x[0]))
    histogram1_1_x = histogram1_1_x[1:] * (perspective_img1.shape[-1]*
                    perspective_img1.shape[-2]/(perspective_img1.shape[-1]*
                    perspective_img1.shape[-2]-histogram1_1_x[0]))
    histogram1_2_x = histogram1_2_x[1:] * (perspective_img1.shape[-1]*
                    perspective_img1.shape[-2]/(perspective_img1.shape[-1]*
                    perspective_img1.shape[-2]-histogram1_2_x[0]))
    # plot 1:
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].plot(histogram1_0_x)
    axarr[0, 1].plot(histogram1_1_x)
    axarr[1, 0].plot(histogram1_2_x)
    axarr[1, 1].imshow(perspective_img1.permute(2, 1, 0))

    plt.suptitle("Histogram5")
    plt.savefig("Histogram5")
    ####################################
    # plot 2:
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].plot(histogram2_0[1:])
    axarr[0, 1].plot(histogram2_1[1:])
    axarr[1, 0].plot(histogram2_2[1:])
    axarr[1, 1].imshow(img2.permute(2, 1, 0)*0.5+0.5)

    plt.suptitle("Histogram6")
    plt.savefig("Histogram6")
    ####################################
    distance_R_x = wasserstein_distance(histogram1_0_x, histogram2_0[1:])
    distance_G_x = wasserstein_distance(histogram1_1_x, histogram2_1[1:])
    distance_B_x = wasserstein_distance(histogram1_2_x, histogram2_2[1:])
    print("After perspective Transformation, Wasserstein Distance R Channel Histogram: ",
          distance_R_x)
    print("After perspective Transformation, Wasserstein Distance G Channel Histogram: ",
          distance_G_x)
    print("After perspective Transformation, Wasserstein Distance B Channel Histogram: ",
          distance_B_x)


if __name__ == "__main__":
    main()