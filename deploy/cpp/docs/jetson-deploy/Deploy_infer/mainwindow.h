#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QString>
#include <QLibrary>

#include "inferthread.h"

// 模型初始化与销毁的接口
typedef void (*InitModel)(const char*,
                          const char*,
                          const char*,
                          const char*,
                          bool , int , char* );
typedef void (*DestructModel)();
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

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

private: // 一些标志位或者数据存储变量
    bool has_Init;  // 是否初始化
    bool doing_Infer; // 是否在推理中
    QString model_Envs[2]; // 运行环境集合
    QString model_Kinds[5]; // 模型类型集合
    bool is_paddlex;  // 特别记录paddlex模型
    bool is_mask;     // 特别记录paddlex下的mask模型
    QString model_Env;  // 当前运行环境
    int old_model_Env; // 上一次的运行环境
    QString model_Kind; // 当前模型类型
    int old_model_Kind; // 上一次的模型类型
    int gpu_Id;  // 当前GPU_ID
    int old_gpu_Id;  // 上一次的GPU_ID

    float det_threshold; // 目标检测的检测阈值
    float old_det_threshold; // 上一次的目标检测的检测阈值
    int infer_Delay; // 连续推理间隔时长
    int old_infer_Delay; // 上一次的连续推理间隔时长

    // 模型文件路径
    QString model_path;
    QString param_path;
    QString config_path;

    // 推理数据路径
    QString img_file;
    QStringList img_files;
    QString video_file;

    // 动态库的链接指针
    QLibrary *inferLibrary;
    // 模型加载与销毁接口
    InitModel initModel;
    DestructModel destructModel;
    // 模型推理接口
    Det_ModelPredict det_ModelPredict;
    Seg_ModelPredict seg_ModelPredict;
    Cls_ModelPredict cls_ModelPredict;
    Mask_ModelPredict mask_ModelPredict;

private:
    InferThread *inferThread;

public: // 一些基本方法
    void Init_SystemState();  // 初始化状态与数据变量
    void Init_SystemShow();  // 初始化可视化按键的状态: 允许使用与禁止使用

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
