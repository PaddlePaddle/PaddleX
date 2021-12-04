#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QString>
#include <QLibrary>

#include "infer_thread.h"

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
    bool has_init_;  // Whether it is initialized
    bool doing_infer_; // Whether it is in reason
    QString model_envs_[2]; // Running environment collection
    QString model_kinds_[5]; // Model type collection
    bool is_paddlex_;  // Special record PADDLEX model
    bool is_mask_;     // Specially recorded Mask model under Paddlex
    QString model_env_;  // Current operating environment
    int old_model_env_; // Last run environment
    QString model_kind_; // Current model type
    int old_model_kind_; // The last model type
    int gpu_id_;  // Current gpu_id
    int old_gpu_id_;  // Last gpu_id

    float det_threshold_; // Target detection detection threshold
    float old_det_threshold_; // The detection threshold of the last target detection
    int infer_delay_; // Continuous reasoning interval
    int old_infer_delay_; // The last continuous reasoning interval

    // Model file path
    QString model_path_;
    QString param_path_;
    QString config_path_;

    // Inferential data path
    QString img_file_;
    QStringList img_files_;
    QString video_file_;

private: // Library and Infer API
    // Link pointer to a dynamic library
    QLibrary *inferlibrary_;
    // Model initialization and destruction API
    InitModel initmodel_;
    DestructModel destructmodel_;
    // Model reasoning API
    Det_ModelPredict det_modelpredict_;
    Seg_ModelPredict seg_modelpredict_;
    Cls_ModelPredict cls_modelpredict_;
    Mask_ModelPredict mask_modelpredict_;

private: // sub thread
    InferThread *inferthread_;

public: // Some basic methods
    void Init_SystemState();  // Initialize state and data variables
    void Init_SystemShow();  // Initialize the state of the visual key: enable and disable

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots: // control slot
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


private slots: // data update slot
    void ImageUpdate(QImage* label1, QImage* label2);
    void Btn_StopAndInfer_StateUpdate(bool stop_state, bool infer_state);
    void CostTimeUpdate(double cost_time);

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
