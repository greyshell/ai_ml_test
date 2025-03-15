import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import numpy as np


def train_cifar10():
    # Print GPU info
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    print("Downloading and preparing CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    # Define a CNN model
    class ResBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = torch.relu(out)
            return out

    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)

            self.layer1 = ResBlock(64, 64)
            self.layer2 = ResBlock(64, 128, stride=2)
            self.layer3 = ResBlock(128, 256, stride=2)
            self.layer4 = ResBlock(256, 512, stride=2)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, 10)

        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    # Create model and move to GPU
    model = SimpleCNN().to(device)
    print(f"Model created and moved to {device}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Training function
    def train(epoch):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        batch_time = AverageMeter()
        end = time.time()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % 20 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(trainloader)} | '
                      f'Loss: {train_loss / (batch_idx + 1):.3f} | '
                      f'Acc: {100. * correct / total:.3f}% | '
                      f'Time: {batch_time.avg:.3f}s')

        return train_loss / len(trainloader), 100. * correct / total

    # Testing function
    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        print(f'Epoch: {epoch} | Test Loss: {test_loss / len(testloader):.3f} | '
              f'Acc: {100. * correct / total:.3f}%')

        return test_loss / len(testloader), 100. * correct / total

    # Helper class for averaging
    class AverageMeter(object):
        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    # Train and test for 5 epochs
    epochs = 5
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    print(f"\nStarting training for {epochs} epochs...")
    total_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        scheduler.step()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds\n")

    total_time = time.time() - total_start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Average time per epoch: {total_time / epochs:.2f} seconds")

    # Save the model
    torch.save(model.state_dict(), 'cifar10_cnn.pth')
    print("Model saved as 'cifar10_cnn.pth'")

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Test Accuracy')

    plt.tight_layout()
    plt.savefig('cifar10_training_results.png')
    plt.show()

    print("Results saved to 'cifar10_training_results.png'")

    return model


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        train_cifar10()
    else:
        print("No GPU available. This test requires a GPU.")