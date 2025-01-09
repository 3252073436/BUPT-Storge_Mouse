本项目主要处理来自本地计算机存储的图像和视频文件。用户通过上传文件夹路径，系统扫描该文件夹并获取其中的所有图像和视频文件。支持的图像文件格式包括PNG、JPG、JPEG、BMP等常见图像格式，支持的视频文件格式包括MP4、AVI、MOV等。



使用前的前提条件：用户必须拥有python环境，在requirements.txt中列出了运行该项目所必须的python库:

·  Flask: 用于构建 web 应用的框架。

·  transformers: 用于加载和使用 ChineseCLIPModel 和 ChineseCLIPProcessor（来自 Hugging Face）。

·  torch: 用于深度学习模型的推理，尤其是用于 ChineseCLIPModel。

·  Pillow: 用于图像处理（比如打开图像和提取图像特征）。

·  numpy: 用于数组处理和计算，相似度计算也需要它。

·  sqlite3: 用于连接 SQLite 数据库。

·  subprocess: 用于调用外部命令，如 ffmpeg。



在项目文件夹下，用户需要运行app.py文件实现核心功能，在第一次使用前需要再改文件内设置模型路径：

无pycharm、VScode等其他编程软件可用记事本打开app.py:

将“r”后的路径改成用户电脑中模型文件夹所在位置（要保留那个“r”），在项目文件夹中，模型默认保存在“项目目录\models\OFA-Syschinese-clip-vit-base-patch16”中

合法的文件夹下应有config.json、preprocessor_config.json、pytorch_model.bin等类似格式文件。


下一步，运行app.py后，打开任意浏览器，进入网址http://127.0.0.1:5000 即可使用主要功能

·  文件夹扫描与图像特征提取
在用户提供文件夹路径后，系统会扫描该文件夹中的所有图像和视频文件，提取其特征并存入数据库。扫描过程的进度通过前端动态更新显示。

·  ·  图像查询结果
用户可以通过文本输入进行图像查询，系统返回与查询文本最相关的图像及其相似度。查询结果展示在表格中，包括图像缩略图、相似度分数和文件原始路径。
·  示例查询结果：
·  查询文本： "初音未来"

返回结果： 系统根据文本特征与数据库中存储的图像特征进行匹配，返回最相似的图像和其相似度，图像可以直接查看。
