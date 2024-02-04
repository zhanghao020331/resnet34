
import glob
import json
import torch
import pandas as pd
from torchvision import transforms
from model import resnet34

import os
from PIL import Image, ImageDraw, ImageFont

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(64),
         transforms.CenterCrop(64),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # 指向需要遍历预测的图像文件夹
    imgs_root = r'D:\hua\x10'
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".png")]

    # read class_indict
    json_path = 'D:/hua/ResNet-pytorch-main/class_index.josn'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=20).to(device)

    # load model weights
    weights_path = r"D:\hua\ResNet-pytorch-main\resnet34-1100.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()

    batch_size = 8  # 每次预测时将多少张图片打包成一个batch
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)


            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(cla.numpy())],
                                                                 pro.numpy()))


    # 类别名称列表
    class_names = ["0","05","1","15","2","25","3","35","4","45","5","55","6","65","7","75","8","85","9","95"]
    results=[]
    # 待处理图像的文件夹路径和保存结果图像的文件夹路径
    input_folder = "D:/hua/x10"
    output_folder = "D:/hua/x10shu"

    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, filename)

                # 打开图像
                image = Image.open(image_path)

                # 预测图像类别和置信度
                input_tensor = data_transform(image).unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)

                _, predicted_idx = torch.max(output, 1)
                confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_idx]
                class_name = class_names[predicted_idx.item()]

                # 创建类别文件夹并保存图像
                output_subfolder = os.path.join(output_folder, class_name)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                # 在图像上打印类别和置信度
                draw = ImageDraw.Draw(image)
                font_size = 15
                font = ImageFont.truetype("arial", font_size)

                confidence_value = confidence.item()  # 将张量值转换为 Python 浮点数

                class_text = f"C:{class_name}"
                class_text_width, class_text_height = font.getbbox(class_text)[2], font.getbbox(class_text)[3]

                confidence_text = f"d:{confidence_value:.2f}"
                confidence_text_width, confidence_text_height = font.getbbox(confidence_text)[2], font.getbbox(confidence_text)[
                    3]

                # 绘制文本框
                x = 1
                y = 1


                # 输出类别
                draw.text((x, y), class_text, fill=(255, 0, 0), font=font)
                y += class_text_height + 30  # 调整每行之间的间距

                # 输出置信度
                draw.text((x, y), confidence_text, fill=(255, 0, 0), font=font)

                # 保存结果图像
                output_path = os.path.join(output_subfolder, filename)
                image.save(output_path)
                print(f"Result image saved at: {output_path}")



                # 在循环中处理每个图像文件时，获取当前文件夹名称
                current_folder_name = os.path.basename(os.path.dirname(image_path))

                # 修改添加到results列表中的字典，包括当前文件夹名称
                result = {
                    'Filename': filename,
                    'Category': class_name,
                    'Confidence': confidence_value,
                    'Current Folder': current_folder_name  # 添加当前文件夹名称
                }
                results.append(result)

    # 创建一个DataFrame对象
    df = pd.DataFrame(results)

    # 将DataFrame保存到Excel文件
    output_file = os.path.join(output_folder, 'classification_results.xlsx')
    df.to_excel(output_file, index=False)
    print(f"Classification results saved at: {output_file}")




if __name__ == '__main__':
    main()
