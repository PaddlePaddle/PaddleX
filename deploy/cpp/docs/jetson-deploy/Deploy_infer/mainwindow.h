#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QString>
#include <QLibrary>

#include "inferthread.h"

// Model initialization and destruction API
typedef void (*InitModel)(const char*,
                          const char*,
                          const char*,
                          const char*,
                          bool , int , char* );
typedef void (*DestructModel)();
// Model reasoning API: det, seg, clas, mask
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

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

private: // Some logo or data storage variables
    bool has_Init;  // Whether it is initialized
    bool doing_Infer; // Whether it is in reason
    QString model_Envs[2]; // Running environment collection
    QString model_Kinds[5]; // Model type collection
    bool is_paddlex;  // Special record PADDLEX model
    bool is_mask;     // Specially recorded Mask model under Paddlex
    QString model_Env;  // Current operating environment
    int old_model_Env; // Last run environment
    QString model_Kind; // Current model type
    int old_model_Kind; // The last model type
    int gpu_Id;  // Current GPU_ID
    int old_gpu_Id;  // Last GPU_ID

    float det_threshold; // Target detection detection threshold
    float old_det_threshold; // The detection threshold of the last target detection
    int infer_Delay; // Continuous reasoning interval
    int old_infer_Delay; // The last continuous reasoning interval

    // Model file path
    QString model_path;
    QString param_path;
    QString config_path;

    // Inferential data path
    QString img_file;
    QStringList img_files;
    QString video_file;

    // Link pointer to a dynamic library
    QLibrary *inferLibrary;
    // Model initialization and destruction API
    InitModel initModel;
    DestructModel destructModel;
    // Model reasoning API
    Det_ModelPredict det_ModelPredict;
    Seg_ModelPredict seg_ModelPredict;
    Cls_ModelPredict cls_ModelPredict;
    Mask_ModelPredict mask_ModelPredict;

private:
    InferThread *inferThread;

public: // Some basic methods
    void Init_SystemState();  // Initialize state and data variables
    void Init_SystemShow();  // Initialize the state of the visual key: enable and disable

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_btnInit_clicked();

    void on_btnDistory_clicked();

    void on_btnLoadImg_clicked();

    void on_btnLoadImgs_clicked();

    void on_btnLoadVideo_clicked();

    void on_btnInfer_clicked();

    void on_btnStop_clicked();

    void on_cBoxEnv_currentIndexChanged(int index);

    void on_cBoxKind_currentIndexChanged(int index);

    void on_sBoxThreshold_valueChanged(double arg1);

    void on_lEditGpuId_textChanged(const QString &arg1);

    void on_sBoxDelay_valueChanged(const QString &arg1);


private slots:
    void ImageUpdate(QImage* label1, QImage* label2);
    void Btn_StopAndInfer_StateUpdate(bool stop_state, bool infer_state);
    void CostTimeUpdate(double cost_time);

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
