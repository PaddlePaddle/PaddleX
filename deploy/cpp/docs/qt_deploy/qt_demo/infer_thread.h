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
typedef void (*DetModelPredict)(const unsigned char* ,
                                 int , int , int ,
                                 float* , int* , char* );
typedef void (*SegModelPredict)(const unsigned char* ,
                                 int , int , int ,
                                 unsigned char* );
typedef void (*ClsModelPredict)(const unsigned char* ,
                                 int , int , int ,
                                 float* , char* , int* );
typedef void (*MaskModelPredict)(const unsigned char* ,
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
    DetModelPredict det_modelpredict_;
    SegModelPredict seg_modelpredict_;
    ClsModelPredict cls_modelpredict_;
    MaskModelPredict mask_modelpredict_;

private:
    // don`t use
    QPushButton *btnstop_;  // Point to the external termination button
    QPushButton *btninfer_;  // Point to the external reasoning button
    QLabel *labelImage1_;   // Image display area-Label--left
    QLabel *labelimage2_;   // Image display area-Label--right

public:
    void set_btn_stop(QPushButton *btn);  // control btn state
    void set_btn_infer(QPushButton *btn);

    void set_det_threshold(float threshold);
    void set_infer_delay(int delay);
    void get_color_map_list(int num_classes=256);

public:
    explicit InferThread(QObject *parent = nullptr);
    void set_model_type(const QString & model_type);  // Setting the model type of reasoning - use the correct model reasoning interface
    void set_input_image(const QString & image_path);
    void set_input_images(const QStringList & images_path);
    void set_input_video(const QString & video_path);
    void set_infer_funcs(DetModelPredict det_inferfunc, SegModelPredict seg_inferfunc,
                       ClsModelPredict cls_inferfunc, MaskModelPredict mask_inferfunc);
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
    void DetImageInfer();
    void DetImagesInfer();
    void DetVideoInfer();

// Semantic segmentation reasoning interface
public:
    void SegImageInfer();
    void SegImagesInfer();
    void SegVideoInfer();

// Classification inference interface
public:
    void ClsImageInfer();
    void ClsImagesInfer();
    void ClsVideoInfer();

// Instance split reasoning interface
public:
    void MaskImageInfer();
    void MaskImagesInfer();
    void MaskVideoInfer();

signals:
    void InferFinished(QImage* label1, QImage* label2);
    void SetStateBtnStopAndInfer(bool stop_state, bool infer_state);
    void SetCostTime(double cost_time);

public slots:
};

#endif // INFERTHREAD_H
