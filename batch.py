
import os
import json

import torch

from torchvision import transforms
from PIL import Image, ImageDraw
from model import resnet34
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 与训练的预处理一样
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载图片
    img_path = 'D:/hua/02/12849.png'
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    image = Image.open(img_path)

    # image.show()
    # [N, C, H, W]
    img = data_transform(image)
    # 扩展维度
    img = torch.unsqueeze(img, dim=0)

    # 获取标签
    json_path = 'D:/hua/ResNet-pytorch-main/class_index.josn'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
    with open(json_path, 'r') as f:
        # 使用json.load()函数加载JSON文件的内容并将其存储在一个Python字典中
        class_indict = json.load(f)

    # 加载网络
    model = resnet34(num_classes=5).to(device)

    # 加载模型文件
    weights_path = r"D:\hua\ResNet-pytorch-main\resnet34-1100.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    with torch.no_grad():
        # 对输入图像进行预测
        output = torch.squeeze(model(img.to(device))).cpu()
        # 对模型的输出进行 softmax 操作，将输出转换为类别概率
        predict = torch.softmax(output, dim=0)
        # 得到高概率的类别的索引
        predict_cla = torch.argmax(predict).numpy()

    res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
    draw = ImageDraw.Draw(image)




    # 文本的左上角位置
    position = (10, 10)
    # fill 指定文本颜色
    draw.text(position, res, fill='red')
    image.show()
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)], predict[i].numpy()))


if __name__ == '__main__':
    main()