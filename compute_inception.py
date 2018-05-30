from utils.inception_score import inception_score, IgnoreLabelDataset
import torchvision.datasets as dset
import torchvision.transforms as transforms


root='databases/mod_cls_gan'
db = dset.ImageFolder(root=root,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

print("Calculating Inception Score...")
print(inception_score(IgnoreLabelDataset(db), cuda=True, batch_size=64,
                      resize=True, splits=10))
