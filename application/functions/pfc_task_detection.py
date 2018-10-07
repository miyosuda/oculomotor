# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        #print(x.size())
        x = x.view(-1, 32 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PFCTaskDetection(object):
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #print(self.device)
        self.classes = ('ppt', 'cd', 'ooo', 'vs', 'mot', 'rdmd')
        self.transform = transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.net = VGGNet()
        #self.net = torchvision.models.vgg11()
        self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        #self.optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        # https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08) # default parameters

    def load_model(self, model_filename):
        return self.net.load_state_dict(torch.load(model_filename, map_location=self.device))

    def save_model(self, model_filename):
        return torch.save(self.net.state_dict(), model_filename)

    def train(self, data_dir, batch_size, max_epoch):
        trainset = torchvision.datasets.ImageFolder(data_dir, transform=self.transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.net.train()
        for epoch in range(max_epoch):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 100 == 99:    # print every 100 mini-batches
                    print('[%d, %5d] loss: %.4f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

    def test(self, data_dir, batch_size):
        testset = torchvision.datasets.ImageFolder(data_dir, transform=self.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        class_correct = list(0. for i in range(len(self.classes)))
        class_total = list(0. for i in range(len(self.classes)))
        self.net.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(batch_size):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        for i in range(len(self.classes)):
            print('Accuracy of %5s : %d / %d, %2.1f %%' % (
                self.classes[i], class_correct[i], class_total[i], 100 * class_correct[i] / class_total[i]))

    def test_image(self, image):
        images = self.transform(image).view(-1, 3, 128, 128) # torch.Size([1, 3, 128, 128])
        self.net.eval()
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.net(images)
            _, predicted = torch.max(outputs, 1)
            i = predicted[0].item()
            return i + 1, self.classes[i]


if __name__ == '__main__':
    pfc_td = PFCTaskDetection()
    #pfc_td.train('./data/train/', 64, 20)
    #pfc_td.save_model('pfc_task_detection.pth')
    pfc_td.load_model('application/functions/data/pfc_task_detection.pth')
    #pfc_td.test('./data/test/', 64)
    #pfc_td.test('../oculoenv/data/test/', 64)
    import matplotlib.pyplot as plt
    for answer_id, answer_name in enumerate(pfc_td.classes, 1):
        for img_id in range(640):
            #filename = './data/test/{}/{}.png'.format(answer_id, img_id)
            filename = '../oculoenv/data/test/{}/{}.png'.format(answer_id, img_id)
            predicted_id, predicted_name = pfc_td.test_image(plt.imread(filename))
            if answer_id != predicted_id:
                print(answer_name, predicted_name, filename)
