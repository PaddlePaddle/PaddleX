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

// Model inference API: det, seg, clas, mask
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
    uchar *color_map;   // visual color table/map

    QString image_path;
    QStringList images_path;
    QString video_path;

public:
    float det_Threshold;  // Target detection threshold
    int infer_Delay;    // Continuous inference interval

    bool doing_Infer;  // Sign reasoning
    bool break_Infer;  // Terminate continuous reasoning/video reasoning

    bool dataLoaded;  // Whether data has been loaded

    QImage* label1_image;
    QImage* label2_image;

    Mat* image1;
    Mat* image2;


private:
    // Model inference API
    Det_ModelPredict det_ModelPredict;
    Seg_ModelPredict seg_ModelPredict;
    Cls_ModelPredict cls_ModelPredict;
    Mask_ModelPredict mask_ModelPredict;

private:
    // don`t use
    QPushButton *btnStop;  // Point to the external termination button
    QPushButton *btnInfer;  // Point to the external reasoning button
    QLabel *labelImage1;   // Image display area-Label--left
    QLabel *labelImage2;   // Image display area-Label--right

public:
    void setStopBtn(QPushButton *btn);
    void setInferBtn(QPushButton *btn);

    void setDetThreshold(float threshold);
    void setInferDelay(int delay);
    uchar* get_color_map_list(int num_classes=256);

public:
    explicit InferThread(QObject *parent = nullptr);
    void setModelType(QString & model_type);  // Setting the model type of reasoning - use the correct model reasoning interface
    void setInputImage(QString & image_path);
    void setInputImages(QStringList & images_path);
    void setInputVideo(QString & video_path);
    void setInferFuncs(Det_ModelPredict det_Inferfunc, Seg_ModelPredict seg_Inferfunc,
                       Cls_ModelPredict cls_Inferfunc, Mask_ModelPredict mask_Inferfunc);
    void runInferDet();
    void runInferSeg();
    void runInferCls();
    void runInferMask();
    void run() override; // Execute this thread

private:
    bool is_InferImage();
    bool is_InferImages();
    bool is_InferVideo();
    QString makeLabelInfo(QString label, int id, float score);

// Detecting the inference interface
public:
    void Det_Image();
    void Det_Images();
    void Det_Video();

// Semantic segmentation reasoning interface
public:
    void Seg_Image();
    void Seg_Images();
    void Seg_Video();

// Classification inference interface
public:
    void Cls_Image();
    void Cls_Images();
    void Cls_Video();

// Instance split reasoning interface
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
