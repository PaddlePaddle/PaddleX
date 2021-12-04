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
    QString model_type_;
    uchar *color_map_;   // visual color table/map

    QString image_path_;
    QStringList images_path_;
    QString video_path_;

public:
    float det_threshold_;  // Target detection threshold
    int infer_delay_;    // Continuous inference interval

    bool doing_infer_;  // Sign reasoning
    bool break_infer_;  // Terminate continuous reasoning/video reasoning

    bool dataloaded_;  // Whether data has been loaded

    QImage* label1_image_;
    QImage* label2_image_;

    Mat* image1_;
    Mat* image2_;


private:
    // Model inference API
    Det_ModelPredict det_modelpredict_;
    Seg_ModelPredict seg_modelpredict_;
    Cls_ModelPredict cls_modelpredict_;
    Mask_ModelPredict mask_modelpredict_;

private:
    // don`t use
    QPushButton *btnstop_;  // Point to the external termination button
    QPushButton *btninfer_;  // Point to the external reasoning button
    QLabel *labelImage1_;   // Image display area-Label--left
    QLabel *labelimage2_;   // Image display area-Label--right

public:
    void setbtnstop(QPushButton *btn);  // control btn state
    void setbtninfer(QPushButton *btn);

    void setdetthreshold(float threshold);
    void setinferdelay(int delay);
    void get_color_map_list(int num_classes=256);

public:
    explicit InferThread(QObject *parent = nullptr);
    void setmodeltype(const QString & model_type);  // Setting the model type of reasoning - use the correct model reasoning interface
    void setinputimage(const QString & image_path);
    void setinputimages(const QStringList & images_path);
    void setinputvideo(const QString & video_path);
    void setinferfuncs(Det_ModelPredict det_inferfunc, Seg_ModelPredict seg_inferfunc,
                       Cls_ModelPredict cls_inferfunc, Mask_ModelPredict mask_inferfunc);
    void RunInferDet();
    void RunInferSeg();
    void RunInferCls();
    void RunInferMask();
    void run() override; // Execute this thread

private:
    bool is_inferimage();
    bool is_inferimages();
    bool is_infervideo();
    QString makelabelinfo(QString label, int id, float score);

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
