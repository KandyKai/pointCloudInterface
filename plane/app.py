# -*- ecoding: utf-8 -*-
# @ModuleName: app
# @Author: Kandy
# @Time: 2023-11-20 17:08
import cv2
from PIL import Image
from flask import Flask, request, send_file
from test_cls import parse_args,main
import io
import os

app = Flask(__name__)

@app.route("/start", methods=['POST','GET'])
def start():
    file_content = request.form['file_content']
    print(file_content)
    return "文件成功接收！！！"


    txtPath = request.form.get("path") # str
    # txtPath = request.args.get('path')
    # true_class_name = txtPath.split('\\')[-2]

    if txtPath == None:
        return "没有接收到地址!!!"
    print("-------------》",txtPath)
    filename = txtPath.split("\\")[-1]  # 使用split方法将路径按照反斜杠进行分割，并取最后一个元素
    true_class_name = filename.split("_")[0]
    args = parse_args()
    pred_class_name = main(args, txtPath)
    print(pred_class_name)
    return pred_class_name
    # return "真实类别:" + true_class_name + ",预测类别:" + pred_class_name


# @app.route("/getHotMap", methods=['POST','GET'])
# def getHotMap():
#     # image_path = request.form.get("path")  # str
#     # print("图片地址:",image_path)
#     image_path = r"C:\Users\Administrator\Desktop\plane\hotpotPic\test.png"
#     # with open(image_path,'r') as file:
#     #     return file
#     # 打开图片文件
#     # 读取图像
#     img = cv2.imread(image_path)
#
#     # 将图像转换成字节流s
#     _, buffer = cv2.imencode('.png', img)
#     byte_img = buffer.tobytes()
#     return send_file(io.BytesIO(byte_img),mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True, port=5000,host='0.0.0.0')

