#ifndef INFERTHREAD_H
#define INFERTHREAD_H

#include <QObject>
#include <QString>
#include <QException>
#include <QDebug>
#include <QPixmap>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QLabel>
#include <QThread>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

// 模型推理接口: det, seg, clas, mask
typedef void (*Det_ModelPredict)(const unsigned char* ,
                                 int , int , int ,
                                 float* , int* , char* );
typedef void (*Seg_ModelPredict)(const unsigned char* ,
                                 int , int , int ,
                                 unsigned char* );
typedef void (*Cls_ModelPredict)(const unsigned char* ,
                                 int , int , int ,
                                 float* , char* , int* );
typedef void (*Mask_ModelPredict)(const unsigned char* ,
                                  int , int , int ,
                                  float* , unsigned char* ,
                                  int* , char* );

class InferThread : public QThread
{
    Q_OBJECT
private:
    QString model_Type;
    uchar *color_map;   // 绘制可视化的颜色表

    QString image_path;
    QStringList images_path;
    QString video_path;

public:
    float det_Threshold;  // 目标检测阈值
    int infer_Delay;    // 连续推理间隔

    bool doing_Infer;  // 标志推理的进行
    bool break_Infer;  // 终止连续推理/视频推理

    bool dataLoaded;  // 是否已加载数据

    QImage* label1_image;
    QImage* label2_image;

    Mat* image1;
    Mat* image2;


private:
    // 模型推理接口
    Det_ModelPredict det_ModelPredict;
    Seg_ModelPredict seg_ModelPredict;
    Cls_ModelPredict cls_ModelPredict;
    Mask_ModelPredict mask_ModelPredict;

private:
    QPushButton *btnStop;  // 指向外部的终止按键
    QPushButton *btnInfer;  // 指向外部的推理按键
    QLabel *labelImage1;   // 图像显示区域Label--左
    QLabel *labelImage2;   // 图像显示区域Label--右

public:
    void setStopBtn(QPushButton *btn);
    void setInferBtn(QPushButton *btn);

    void setDetThreshold(float threshold);
    void setInferDelay(int delay);
    uchar* get_color_map_list(int num_classes=256);

public:
    explicit InferThread(QObject *parent = nullptr);
    void setModelType(QString & model_type);  // 设置推理的模型类型 -- 使用正确的模型推理接口
    void setInputImage(QString & image_path);
    void setInputImages(QStringList & images_path);
    void setInputVideo(QString & video_path);
    void setInferFuncs(Det_ModelPredict det_Inferfunc, Seg_ModelPredict seg_Inferfunc,
                       Cls_ModelPredict cls_Inferfunc, Mask_ModelPredict mask_Inferfunc);
    void runInferDet();
    void runInferSeg();
    void runInferCls();
    void runInferMask();
    void run() override; // 执行该线程

private:
    bool is_InferImage();
    bool is_InferImages();
    bool is_InferVideo();
    QString makeLabelInfo(QString label, int id, float score);

// 检测推理接口
public:
    void Det_Image();
    void Det_Images();
    void Det_Video();

// 语义分割推理接口
public:
    void Seg_Image();
    void Seg_Images();
    void Seg_Video();

// 分类推理接口
public:
    void Cls_Image();
    void Cls_Images();
    void Cls_Video();

// 实例分割推理接口
public:
    void Mask_Image();
    void Mask_Images();
    void Mask_Video();

signals:
    void InferFinished(QImage* label1, QImage* label2);
    void SetState_Btn_StopAndInfer(bool stop_state, bool infer_state);
    void SetCostTime(double cost_time);

public slots:
};

#endif // INFERTHREAD_H
