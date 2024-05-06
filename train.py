import torch
from train_common import *
import utils
from model import Model
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from dataset import load_data


# def freeze_layers(model, num_layers=0):
#     for name, param in model.named_parameters():
#         if name[0] == 'c' and int(name[4]) <= num_layers:
#             param.requires_grad = False


def train(tr_loader, va_loader, te_loader, model, model_name, num_layers=0):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())

    print("Loading target model with", num_layers, "layers frozen")
    model, start_epoch, stats = restore_checkpoint(model, model_name)

    axes = utils.make_training_plot("Target Training")

    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        include_test=True,
        multiclass=True,
    )

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while epoch < 5:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
            include_test=True,
            multiclass=True,
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, model_name, stats)
        epoch += 1

    print("Training finished.")

    utils.save_tl_training_plot(num_layers)
    utils.hold_training_plot()


def test(m):
    test_letters = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6,
                    'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13,
                    'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19,
                    'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25,
                    'del': 26, 'nothing': 27, 'space': 28}
    fig, axs = plt.subplots(4, 7, figsize=(10, 8))

    axs = axs.flatten()

    for idx, letter in enumerate(test_letters):
        if letter == 'del': # there is no del test image in our testing dataset.
            continue
        if letter in ['space', 'nothing']:
            idx -= 1
        image = Image.open(f'./asl_alphabet_test/{letter}_test.jpg')

        preprocess = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        image_tensor = preprocess(image)
        image_tensor = image_tensor.unsqueeze(0)
        out = m(image_tensor)

        prediction = predictions(out.data)
        prediction_letter = ''
        for key, value in test_letters.items():
            if prediction.item() == value:
                prediction_letter = key
        axs[idx].imshow(image_tensor[0].permute(1, 2, 0) * 0.5 + 0.5)
        axs[idx].set_title(f"True: {letter}\nPred: {prediction_letter}")
        axs[idx].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # Data loaders
    tr_loader, va_loader, te_loader, _ = load_data() # Comment this out if you don't want to train
    m = Model()

    """Regular training model. No data modification (5 epochs)"""
    train(tr_loader, va_loader, te_loader, m, 'model1', 0) # Comment this out if you don't want to train
    # m, _, _ = restore_checkpoint(m, './model1') # Comment this out if you're training

    """Training model with data modification (5 epochs)"""
    # train(tr_loader, va_loader, te_loader, m, 'model2', 0) # Comment this out if you don't want to train
    # m, _, _ = restore_checkpoint(m, './model2') # Comment this out if you're training

    test(m)


if __name__ == "__main__":
   main()
