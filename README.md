## 进行人脸比对（三步骤）
1. 人脸检测、活体检测
2. 人脸对齐（关键点检测：dlib）
3. 人脸比对

## 两个应用场景:
1. 1:N---考勤系统
2. 1:1---身份验证

## 活体检测
二分类
可以利用rgb，深度图像，红外三种concat在一起识别

## 数据集
1. 活体检测数据集
链接：https://pan.baidu.com/s/1-RDeHj0Z9bAVQzX1xrNU8Q  提取码：ay1y 
CASIA-SURF数据集，包括了录制的视频、train、test、valid及其相应的标签。

2. 人脸识别数据集
在lfw数据集上进行准确率测试

## 模型
1. FaceBagNet (用于活体检测)
    Patch-based feature:块特征
    Multi-stream fusion with MFE:多模态擦除式融合
2. FaceNet （用于人脸识别）
    triplet loss：三元组
    困难样本

## 学习率
周期余弦退火

## 优化
剪裁卷积核
模型压缩：训练完成后再裁减
基于L1 norm 的模型压缩