# yolo3_reid
环境pytorch1.5+cuda10.1，pycharm


克隆这个项目，同时新建几个文件夹，完整目录如下：

input_video
output
query
reid
weights
yolo3
hst_search.py
search.py
query_get.py

input_video存放待检测的视频或图片或came
检测结果会输出到output文件夹
query输入要查询的人物截图
weights存放yolov3.weights和reid.pth
hst_search.py 是利用颜色直方图检索行人ID
search.py是基于strong_baseline的深度学习检索方法。

运行步骤：
1。按照上面结构新建input_video,output,weights文件夹。
2.下载yolov3.weights权重和reid.pth放入weights文件夹，下载链接：
3.将要检索的视频放入input_video文件夹。
4.通过query_get.py截图要检索的人物，截取的图片自动放入query中
5.python search.py 或者 hst_search.py
