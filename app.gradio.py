import gradio as gr
import numpy as np
import cv2
import fastdeploy 
import fastdeploy.vision as vision

# 百度飞桨模型
# L
def infer1(img, score_threshold):
    net = vision.detection.PPYOLOE(
    "./ppyoloe_l_80e/output_inference/model.pdmodel",
    "./ppyoloe_l_80e/output_inference/model.pdiparams", 
    "./ppyoloe_l_80e/output_inference/infer_cfg.yml" )
    org_img = net.predict(img)
    org_img = vision.vis_detection(img, org_img,labels=["normal","falling","fall","sit"],score_threshold=score_threshold,font_color=[0, 0, 205],line_size=3,font_size=1)
    return org_img

# M
def infer2(img, score_threshold):
    net2 = vision.detection.PPYOLOE(
    "./ppyoloe_m_80e/output_inference/model.pdmodel",
    "./ppyoloe_m_80e/output_inference/model.pdiparams", 
    "./ppyoloe_m_80e/output_inference/infer_cfg.yml" )
    org_img = net2.predict(img)
    org_img = vision.vis_detection(img, org_img,labels=["normal","falling","fall","sit"],score_threshold=score_threshold,font_color=[0, 0, 205],line_size=3,font_size=1)
    return org_img

# S
def infer3(img, score_threshold):
    net3 = vision.detection.PPYOLOE(
    "./ppyoloe_s_80e/output_inference/model.pdmodel",
    "./ppyoloe_s_80e/output_inference/model.pdiparams", 
    "./ppyoloe_s_80e/output_inference/infer_cfg.yml" )
    org_img = net3.predict(img)
    org_img = vision.vis_detection(img, org_img,labels=["normal","falling","fall","sit"],score_threshold=score_threshold,font_color=[0, 0, 205],line_size=3,font_size=1)
    return org_img

# 模型选择函数
def generate_image(model, img, score_threshold):
    if model == 'PP-YOLOE-L':
        result = infer1(img, score_threshold)
    elif model == 'PP-YOLOE-M':
        result = infer2(img, score_threshold)
    elif model == 'PP-YOLOE-S':
        result = infer3(img, score_threshold)
    return result

demo = gr.Blocks()

with demo:
    gr.Markdown(
        r"""
        # <center> 🤗 跌倒行为检测 🤗</center>
        """
    )
    gr.Markdown(
        r"""
        
        ## <left> 模型 👉 ``PP-YOLOE-L（暂时不可用） & PP-YOLOE-M（暂时不可用） & PP-YOLOE-S`` ,根据需求选择实时目标检测模型。 </left>

        ## <left> 标签解释：0代表normal(正常)，1代表falling(准备跌倒)，2代表fall(已经跌倒)，3代表sit(坐着) </left>

        ## <left> 请在运行之前选择您想使用的model，否则程序无法正常运行！ </left>

        ## <left> 团队成员：黄德攒、卢春锦、梁磊琳 </left>
        """
    )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                model = gr.Dropdown(["PP-YOLOE-L","PP-YOLOE-M","PP-YOLOE-S"], label="Model")
            with gr.Row():
                threshold = gr.Slider(0.0, 1.0, value=0.4, step=0.01, label="score_threshold")
        with gr.Column():
            with gr.Row():
                inputs = gr.Image(label="Image")
            with gr.Row():
                btn1 = gr.Button(value="🍧Run🍧")
                btn2 = gr.Button(value="🍀Clear🍀")
            with gr.Row():
                output = gr.Image()
            with gr.Row():
                btn3 = gr.Button(value="✨save result✨")

    btn1.click(generate_image, [model, inputs, threshold], output)

demo.launch()

