from nets import Total_Net
import torch
import os
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image


class Trainer:
    def __init__(self, save_net_path, net_name, dataset_path, image_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Total_Net().to(self.device)
        self.net_path = save_net_path
        self.net_name = net_name
        self.image_path = image_path
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.train_data = DataLoader(datasets.MNIST(dataset_path, train=True, transform=self.trans, download=False),
                                     batch_size=100, shuffle=True)
        self.loss_fn = nn.MSELoss(reduction="sum")
        self.optimizer = torch.optim.Adam(self.net.parameters())
        if not os.path.exists(save_net_path):
            os.makedirs(save_net_path)
        else:
            self.net.load_state_dict(torch.load(os.path.join(save_net_path, net_name)))
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        self.net.train()

    def train(self):
        epoch = 1
        loss_new = 100000000
        while True:
            for i, (image, label) in enumerate(self.train_data):
                image = image.to(self.device)
                # 生成标准正太分布的数据
                distribution = torch.randn(128).to(self.device)
                miu, log_sigma, output = self.net(image, distribution)
                encoder_loss = torch.mean((-torch.log(log_sigma ** 2) + miu ** 2 + log_sigma ** 2 - 1) * 0.5)
                decoder_loss = self.loss_fn(output, image)
                loss = encoder_loss + decoder_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if i % 10 == 0:
                    print("epoch:{},i:{},loss:{}".format(epoch, i, loss.item()))
                    image = image.detach()
                    output = output.detach()
                    save_image(image, "{}/{}-{}-input_image.jpg".format(self.image_path, epoch, i), nrow=10)
                    save_image(output, "{}/{}-{}-ouput_image.jpg".format(self.image_path, epoch, i), nrow=10)

                if loss.item() < loss_new:
                    loss_new = loss.item()
                    torch.save(self.net.state_dict(), os.path.join(self.net_path, self.net_name))
            epoch += 1


if __name__ == '__main__':
    trainer = Trainer("models/", "net_sum_pth", "datasets/", "images")
    # trainer = Trainer("models/", "net_mean_pth", "datasets/", "images_mean")
    trainer.train()
