# 欢迎体验本项目
```
团队成员：黄德攒

声明：本项目遵循MIT开源协议，项目内所有文件均可做学术学习使用，禁止任何人以及组织将本项目用作商业用途
```
```
项目介绍：
根目录为./launch，app.gradio.py为demo网页展示启动文件，./ppyoloe_s_80e目录下存放模型文件，requirements.txt为环境所需依赖，根目录下的图片为测试图片。
考虑到多数人的电脑性能不高，所以本项目所存放的模型为small模型，请在进行模型推理的时候注意设备散热。
```
# 模型所能检测的状态
- normal    正常
- falling   即将跌倒
- fall      已经跌倒
- sit       坐姿

其中由于使用者图片角度问题可能会导致状态检测失误或者其它奇奇怪怪的问题，请见谅。

# 使用方法：

1.先准备python环境，python版本需要大于等于3.7，若未具备这方面的知识请自行度娘如何操作。

2.安装环境依赖（若有虚拟环境请注意环境启用情况）
```shell
pip install -r requirements.txt
```
3.启动主应用文件app.gradio.py
```shell
(venv)powershell:python ./app.gradio.py
```

# 注意：
如果需要开启外网访问需要根据系统cmd提示添加参数，项目默认关闭外网访问。

__警告：切勿使用VPN进行翻墙建站操作，由此产生的后果与项目成员无关__

