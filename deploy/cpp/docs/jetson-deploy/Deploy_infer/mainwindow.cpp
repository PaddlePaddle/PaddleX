#include <QMessageBox>
#include <QException>
#include <QFileDialog>
#include <QPixmap>
#include <QDebug>

#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

void MainWindow::Init_SystemState()
{
    qDebug() << "Now Using Opencv Version: " << CV_VERSION << "\n";
//    qDebug() << cv::getBuildInformation().c_str() << "\n"; // out build infomation

    has_Init = false;  // 初始化默认状态为false
    doing_Infer = false; // 推理状态默认为false
    is_paddlex = false;
    is_mask = false;

    model_Envs[0] = "cpu";  // 初始化支持的运行环境列表
    model_Envs[1] = "gpu";
    model_Kinds[0] = "det";  // 初始化支持的模型列表
    model_Kinds[1] = "seg";
    model_Kinds[2] = "clas";
    model_Kinds[3] = "mask";
    model_Kinds[4] = "paddlex";

    model_Env = "cpu";  // 初始化当前的模型状态
    model_Kind = "det";
    gpu_Id = 0;

    det_threshold = 0.5;  // 初始化检测阈值
    infer_Delay = 50;   // 初始化连续推理间隔时长

    ui->cBoxEnv->setCurrentIndex(0); // 初始化环境为CPU
    ui->cBoxKind->setCurrentIndex(0); // 初始化类型为Det
    ui->labelImage1->setStyleSheet("background-color:white;"); // 设置初始化图片展示区域的背景
    ui->labelImage2->setStyleSheet("background-color:white;");

    // 动态链接库的启动
    inferLibrary = new QLibrary("/home/nvidia/Desktop/Deploy_infer/infer_lib/libmodel_infer");
    if(!inferLibrary->load()){
        //加载so失败
        qDebug() << "Load libmodel_infer.so is failed!";
        qDebug() << inferLibrary->errorString();
    }
    else
    {
        qDebug() << "Load libmodel_infer.so is OK!";
    }

    // 引出动态库中的模型加载/销毁接口
    initModel = (InitModel)inferLibrary->resolve("InitModel");
    destructModel = (DestructModel)inferLibrary->resolve("DestructModel");
    // 引出动态图中的模型推理接口
    det_ModelPredict = (Det_ModelPredict)inferLibrary->resolve("Det_ModelPredict");
    seg_ModelPredict = (Seg_ModelPredict)inferLibrary->resolve("Seg_ModelPredict");
    cls_ModelPredict = (Cls_ModelPredict)inferLibrary->resolve("Cls_ModelPredict");
    mask_ModelPredict = (Mask_ModelPredict)inferLibrary->resolve("Mask_ModelPredict");

    // 线程初始化 -- 配置推理函数
    inferThread = new InferThread(this);
    inferThread->setInferFuncs(det_ModelPredict, seg_ModelPredict, cls_ModelPredict, mask_ModelPredict);
    inferThread->setStopBtn(ui->btnStop);
    inferThread->setInferBtn(ui->btnInfer);
    inferThread->setDetThreshold(det_threshold);
    inferThread->setInferDelay(infer_Delay);

    // 配置信号与槽
    connect(inferThread, SIGNAL(InferFinished(QImage*, QImage*)),
            this, SLOT(ImageUpdate(QImage*, QImage*)),
            Qt::BlockingQueuedConnection);
    connect(inferThread, SIGNAL(SetState_Btn_StopAndInfer(bool , bool )),
            this, SLOT(Btn_StopAndInfer_StateUpdate(bool , bool )),
            Qt::BlockingQueuedConnection);
    connect(inferThread, SIGNAL(SetCostTime(double )),
            this, SLOT(CostTimeUpdate(double )),
            Qt::BlockingQueuedConnection);
}

void MainWindow::Init_SystemShow()
{
    ui->btnDistory->setEnabled(false);  // 一开始没有初始化模型，销毁按钮失效
    ui->btnInfer->setEnabled(false);    // 一开始没有初始化模型，推理按钮失效
    ui->btnStop->setEnabled(false);     // 一开始没有推理模型，终止按钮失效
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    this->Init_SystemState(); // 启动时初始化状态
    this->Init_SystemShow();  // 启动时初始化按钮状态
}

MainWindow::~MainWindow()
{
    // 等待线程结束
    if (inferThread->doing_Infer)
    {
        inferThread->break_Infer=false;
        while(inferThread->doing_Infer==true); // 等待线程停止
    }
    // 销毁模型
    if (has_Init == true)
    {
        destructModel();
    }
    //卸载库
    inferLibrary->unload();

    delete ui;  // 最后关闭界面
}

// 模型初始化
// 待补充模型类型的转换
void MainWindow::on_btnInit_clicked()
{
    QString dialog_title = "加载模型";
    QStringList filters = {"*.pdmodel", "*.pdiparams", "*.yml", "*.yaml"};
    QDir model_dir(QFileDialog::getExistingDirectory(this,
                                                     dialog_title));
    QFileInfoList model_files = model_dir.entryInfoList(filters);  // 滤波得到的所有有效文件名

    switch (model_files.count()) {
    case 0:
        return;
    case 3:  // det，seg，clas的加载时可能遇到的情况
        for (int i = 0; i < 3; i++)
        {
            QString tag = model_files[i].fileName().split('.')[1];
            if (tag == "pdmodel") model_path = model_files[i].filePath();
            else if (tag == "pdiparams") param_path = model_files[i].filePath();
            else if (tag == "yml" || tag == "yaml") config_path = model_files[i].filePath();
        }
        break;
    case 4: // paddlex和mask可能遇到的情况
        for (int i = 0; i < 4; i++)
        {
            QString tag = model_files[i].fileName().split('.')[1];
            if (tag == "pdmodel") model_path = model_files[i].filePath();
            else if (tag == "pdiparams") param_path = model_files[i].filePath();
            else if (tag == "yml" || tag == "yaml")
            {
                if (model_files[i].fileName() == "model.yml" || model_files[i].fileName() == "model.yaml")
                    config_path = model_files[i].filePath();
            }
        }
        break;
    default: // 其他未定义的情况
        QMessageBox::information(this,
            tr("提示"),
            tr("请保证模型文件夹下正确包含以下文件:\n*.pdmodel, *.pdiparams, *.yml/*.yaml."),
            QMessageBox::Ok,
            QMessageBox::Ok);
        return;
    }

    char paddlex_model_type[10]="";
    try
    {
        if (has_Init == true)
        {
            destructModel();  // 销毁模型，再初始化
            if (model_Kind == "paddlex")  // paddlex模型特别标识
            {
                is_paddlex = true;
            }
            if (model_Kind == "mask")  // 实例分割来自paddlex
            {
                model_Kind = "paddlex";
                is_mask = true;
            }

            // input string to char*/[]
            initModel(model_Kind.toLocal8Bit().data(),
                      model_path.toLocal8Bit().data(),
                      param_path.toLocal8Bit().data(),
                      config_path.toLocal8Bit().data(),
                      model_Env=="gpu" ? true: false,
                      gpu_Id, paddlex_model_type);

            if (is_paddlex && is_mask==false)  // 替换paddlex模型的实际类型
            {
                model_Kind = QString::fromLocal8Bit(paddlex_model_type);  // to real type
                is_paddlex = false;
            }
            if (is_paddlex==false && is_mask)  // 换回mask的类型
            {
                model_Kind = "mask";  // to mask type
                is_mask = false;
            }
        }
        else
        {
            if (model_Kind == "paddlex")  // paddlex模型特别标识
            {
                is_paddlex = true;
            }
            if (model_Kind == "mask")  // 实例分割来自paddlex
            {
                model_Kind = "paddlex";
                is_mask = true;
            }

            // input string to char*/[]
            initModel(model_Kind.toLocal8Bit().data(),
                      model_path.toLocal8Bit().data(),
                      param_path.toLocal8Bit().data(),
                      config_path.toLocal8Bit().data(),
                      model_Env=="gpu" ? true: false,
                      gpu_Id, paddlex_model_type);

            if (is_paddlex && is_mask==false)  // 替换paddlex模型的实际类型
            {
                model_Kind = QString::fromLocal8Bit(paddlex_model_type);  // to real type
                is_paddlex = false;
            }
            if (is_paddlex==false && is_mask)  // 换回mask的类型
            {
                model_Kind = "mask";  // to mask type
                is_mask = false;
            }

            has_Init = true; // 已经完成初始化了
        }
    }
    catch (QException &e) // 未初始化成功的提示
    {
        QMessageBox::information(this,
            tr("初始化失败"),
            tr("1.请保证模型文件夹下正确包含以下文件:\n*.pdmodel, *.pdiparams, *.yml/*.yaml.\n2.请确保模型类型与加载模型一致."),
            QMessageBox::Ok,
            QMessageBox::Ok);

        if (is_paddlex)  // paddlex未初始化成功，还原类型
        {
            model_Kind = "paddlex";
            is_paddlex = false;
        }
        if (is_mask)  // mask未初始化成功，还原类型
        {
            model_Kind = "mask";
            is_mask = false;
        }

        return;
    }

    inferThread->setModelType(model_Kind); // 设置推理接口类型
    // 初始化成功的提示
    QMessageBox::information(this,
        tr("初始化成功"),
        QString("模型类型: ")+model_Kind+QString(", 运行环境: ")+model_Env+QString("."),
        QMessageBox::Ok,
        QMessageBox::Ok);
    ui->btnInit->setText("模型已初始化");

    ui->btnInfer->setEnabled(true);  // 开启推理功能
    ui->btnDistory->setEnabled(true); // 开启销毁功能
    ui->cBoxEnv->setEnabled(false);   // 关闭运行环境的选择
    ui->cBoxKind->setEnabled(false);   // 关闭运行类型的选择
    ui->lEditGpuId->setEnabled(false);  // 关闭gpu指定
}

// 模型销毁
void MainWindow::on_btnDistory_clicked()
{
    if (inferThread->doing_Infer)  // 正在推理，则无法销毁，发出提示
    {
        QMessageBox::information(this,
            tr("提示"),
            tr("模型正在推理，无法销毁!\n(请推理完成/终止后，再进行模型销毁.)"),
            QMessageBox::Ok,
            QMessageBox::Ok);
        return;
    }

    if (has_Init == true)
    {
        destructModel(); // 销毁模型
        has_Init = false; // 还原初始化状态

        QMessageBox::information(this,
            tr("提示"),
            tr("模型已销毁."),
            QMessageBox::Ok,
            QMessageBox::Ok);
        ui->btnInit->setText("初始化模型");

        ui->btnInfer->setEnabled(false);  // 关闭推理功能
        ui->btnDistory->setEnabled(false); // 关闭销毁功能
        ui->cBoxEnv->setEnabled(true);   // 开启运行环境的选择
        ui->cBoxKind->setEnabled(true);   // 开启运行类型的选择
        ui->lEditGpuId->setEnabled(true);  // 开启gpu指定
    }
    else
    {
        QMessageBox::information(this,
            tr("提示"),
            tr("未初始化，无需销毁."),
            QMessageBox::Ok,
            QMessageBox::Ok);
    }
}

// 加载图片
void MainWindow::on_btnLoadImg_clicked()
{
    QString dialog_title = "加载图片";
    QString filters = "加载图片(*.jpg *.jpeg *.png *.JPEG);;";
    QUrl img_path = QFileDialog::getOpenFileUrl(this,
                                                   dialog_title, QUrl(), filters);
    if (img_path.isEmpty())
    {
        return;
    }

    img_file = img_path.url().split("//")[1]; // 得到图片路径
    qDebug() << "Input Video Path:" << img_file << "\n";

    // 图片读取
    cv::Mat image = cv::imread(img_file.toLocal8Bit().toStdString());  //BGR
    cv::cvtColor(image, image, COLOR_BGR2RGB);  // BGR --> RGB
    cv::resize(image, image, cv::Size(image.cols/4*4,image.rows/4*4)); // 保证pixmap显示正常 -- 切缩放图像
    QImage image_from_mat((const uchar*)image.data, image.cols, image.rows, QImage::Format_RGB888);
    QPixmap pixmap(QPixmap::fromImage(image_from_mat));

    // 显示图片
    ui->labelImage1->setPixmap(pixmap);
    ui->labelImage1->setScaledContents(true);  // 铺满整个label

    inferThread->setInputImage(img_file);  // 押入到推理数据中
    img_files = QStringList();
    video_file = "";

    ui->btnLoadImg->setText("图片已加载");
    ui->btnLoadImgs->setText("加载文件夹");
    ui->btnLoadVideo->setText("加载视频");
}

// 加载图片文件夹
void MainWindow::on_btnLoadImgs_clicked()
{
    QString dialog_title = "加载图片文件夹";
    QStringList filters = {"*.jpg", "*.jpeg", "*.png", "*.JPEG"};
    QDir img_dir(QFileDialog::getExistingDirectory(this,
                                                     dialog_title));
    QFileInfoList img_paths = img_dir.entryInfoList(filters);  // 滤波得到的所有图片有效路径

    if (img_paths.isEmpty())
    {
        return;
    }

    img_files.clear();
    for (int i = 0; i < img_paths.count(); i++)
    {
        img_files.append(img_paths[i].filePath());
    }

    qDebug() << img_files[0] << "\n";

    // 显示第一张图片
    cv::Mat image = cv::imread(img_files[0].toLocal8Bit().toStdString());  //BGR
    cv::cvtColor(image, image, COLOR_BGR2RGB);  // BGR --> RGB
    cv::resize(image, image, cv::Size(image.cols/4*4,image.rows/4*4)); // 保证pixmap显示正常 -- 切缩放图像
    QImage image_from_mat((const uchar*)image.data, image.cols, image.rows, QImage::Format_RGB888);
    QPixmap pixmap(QPixmap::fromImage(image_from_mat));

    // 显示图片
    ui->labelImage1->setPixmap(pixmap);
    ui->labelImage1->setScaledContents(true);  // 铺满整个label

    inferThread->setInputImages(img_files);
    img_file = "";
    video_file = "";

    ui->btnLoadImg->setText("加载图片");
    ui->btnLoadImgs->setText("文件夹已加载");
    ui->btnLoadVideo->setText("加载视频");
}

// 加载视频流
void MainWindow::on_btnLoadVideo_clicked()
{
    QString dialog_title = "加载视频流";
    QString filters = "视频(*.mp4 *.MP4);;";
    QUrl video_path = QFileDialog::getOpenFileUrl(this,
                                                   dialog_title, QUrl(), filters);
    if (video_path.isEmpty())
    {
        return;
    }

    video_file = video_path.url().split("//")[1]; // 得到视频路径
    qDebug() << "Input Video Path:" << video_file.toStdString().c_str() << "\n";

    // 显示视频的第一张图片
    VideoCapture capture;
    Mat frame;
    capture.open(video_file.toLocal8Bit().toStdString()); // 读取视频
    if(!capture.isOpened())
    {
        QMessageBox::information(this,
            tr("提示"),
            tr("1.视频读取失败，请检查视频是否完整，是否为mp4.\n2.(Maybe)Opencv doesn't support the video!"),
            QMessageBox::Ok,
            QMessageBox::Ok);
        return;
    }
    capture >> frame;  // 获取第一帧
    cvtColor(frame, frame, COLOR_BGR2RGB);  // BGR --> RGB
    QImage image((const uchar*)frame.data, frame.cols, frame.rows, QImage::Format_RGB888);
    ui->labelImage1->setPixmap(QPixmap::fromImage(image));  // 加载到label上
    ui->labelImage1->setScaledContents(true);  // 铺满整个label
    capture.release();

    inferThread->setInputVideo(video_file);
    img_file = "";
    img_files = QStringList();

    ui->btnLoadImg->setText("加载图片");
    ui->btnLoadImgs->setText("加载文件夹");
    ui->btnLoadVideo->setText("视频已加载");
}

// 模型推理 -- 会用到多线程
void MainWindow::on_btnInfer_clicked()
{
    if (inferThread->dataLoaded != true)  // 数据未加载
    {
        QMessageBox::information(this,
            tr("提示"),
            tr("请加载数据后，再进行推理."),
            QMessageBox::Ok,
            QMessageBox::Ok);
        return;
    }
    if (has_Init == true && inferThread->doing_Infer == false)
    {
        ui->btnStop->setEnabled(true);     // 终止按钮启动
        ui->btnInfer->setEnabled(false);     // 推理按钮关闭

        // 根据前边读取的数据
        // 加载的模型
        // 执行对应类型的推理
//        inferThread->run(); // 执行推理
        inferThread->start(); // 执行推理
        QMessageBox::information(this,
            tr("提示"),
            tr("模型推理"),
            QMessageBox::Ok,
            QMessageBox::Ok);
    }
}

// 推理终止
void MainWindow::on_btnStop_clicked()
{
    if (inferThread->doing_Infer == true)
    {
        QMessageBox::information(this,
            tr("提示"),
            tr("已终止模型推理"),
            QMessageBox::Ok,
            QMessageBox::Ok);

        inferThread->break_Infer = true;  // 终止推理 --> 自动转到doing_Infer==false
    }
    else
    {
        QMessageBox::information(this,
            tr("提示"),
            tr("未进行推理，无需终止模型推理"),
            QMessageBox::Ok,
            QMessageBox::Ok);
    }
}

// 选择模型运行环境
void MainWindow::on_cBoxEnv_currentIndexChanged(int index)
{
    if (has_Init) // 已经初始化，该操作无效
    {
        ui->cBoxEnv->setCurrentIndex(old_model_Env);
        return;
    }
    model_Env = model_Envs[index];
    old_model_Env = index;  // 保留本次结果
}

// 设置模型类型
void MainWindow::on_cBoxKind_currentIndexChanged(int index)
{
    if (has_Init) // 已经初始化，该操作无效
    {
        ui->cBoxKind->setCurrentIndex(old_model_Kind);
        return;
    }
    model_Kind = model_Kinds[index];
    old_model_Kind = index;
}

// 设置检测阈值
void MainWindow::on_sBoxThreshold_valueChanged(double arg1)
{
    if (inferThread->doing_Infer)
    {
        ui->sBoxThreshold->setValue(old_det_threshold);
        return;
    }
    det_threshold = (float)arg1;
    inferThread->setDetThreshold(det_threshold);

    old_det_threshold = det_threshold;
}

// 设置gpu_id
void MainWindow::on_lEditGpuId_textChanged(const QString &arg1)
{
    if (has_Init) // 已经初始化，该操作无效
    {
        ui->lEditGpuId->setText(QString::number(old_gpu_Id));
        return;
    }
    gpu_Id = arg1.toInt(); // 如果输入为正常数字，则解析为指定的数字；否则为0
    old_gpu_Id = gpu_Id;
}

// 连续推理间隔时长设置
void MainWindow::on_sBoxDelay_valueChanged(const QString &arg1)
{
    if (inferThread->doing_Infer)
    {
        ui->sBoxDelay->setValue(old_infer_Delay);
        return;
    }
    infer_Delay = arg1.toInt(); // 如果输入为正常数字，则解析为指定的数字；否则为0
    inferThread->setInferDelay(infer_Delay);

    old_infer_Delay = infer_Delay;
}


/*  slot funcs blob  */
// update image show
void MainWindow::ImageUpdate(QImage* label1, QImage* label2)
{
    if (inferThread->doing_Infer)
    {
        ui->labelImage1->clear();
        ui->labelImage1->setPixmap(QPixmap::fromImage(*label1));
        ui->labelImage1->setScaledContents(true); // 铺满整个label

        ui->labelImage2->clear();
        ui->labelImage2->setPixmap(QPixmap::fromImage(*label2));
        ui->labelImage2->setScaledContents(true); // 铺满整个label
    }
}

// update btn Enable_state
void MainWindow::Btn_StopAndInfer_StateUpdate(bool stop_state, bool infer_state)
{
    ui->btnStop->setEnabled(stop_state);
    ui->btnInfer->setEnabled(infer_state);
}

// update label value to show cost time
void MainWindow::CostTimeUpdate(double cost_time)
{
    ui->labelCostTime->setText(QString::number(cost_time));
}
