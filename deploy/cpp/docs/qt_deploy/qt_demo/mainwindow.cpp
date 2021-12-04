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
    // out opencv build information -- check ffmpeg whether is ok
    // qDebug() << cv::getBuildInformation().c_str() << "\n";

    has_init_ = false;
    doing_infer_ = false;
    is_paddlex_ = false;
    is_mask_ = false;

    model_envs_[0] = "cpu";
    model_envs_[1] = "gpu";
    model_kinds_[0] = "det";
    model_kinds_[1] = "seg";
    model_kinds_[2] = "clas";
    model_kinds_[3] = "mask";
    model_kinds_[4] = "paddlex";

    model_env_ = "cpu";
    model_kind_ = "det";
    gpu_id_ = 0;

    det_threshold_ = 0.5;
    infer_delay_ = 50;

    ui->cBoxEnv->setCurrentIndex(0); // Initialize the environment to CPU
    ui->cBoxKind->setCurrentIndex(0); // Initialization type Det
    ui->labelImage1->setStyleSheet("background-color:white;"); // Set the background to initialize the image display area
    ui->labelImage2->setStyleSheet("background-color:white;");

    // Dynamic link library startup
    inferlibrary_ = new QLibrary("/home/nvidia/Desktop/Deploy_infer/infer_lib/libmodel_infer");
    if(!inferlibrary_->load()){
        //Loading failed
        qDebug() << "Load libmodel_infer.so is failed!";
        qDebug() << inferlibrary_->errorString();
    }
    else
    {
        qDebug() << "Load libmodel_infer.so is OK!";
    }

    // Export model loading / destruction interface in dynamic library
    initmodel_ = (InitModel)inferlibrary_->resolve("InitModel");
    destructmodel_ = (DestructModel)inferlibrary_->resolve("DestructModel");
    // Export A model insertion interface in dynamic graph
    det_modelpredict_ = (Det_ModelPredict)inferlibrary_->resolve("Det_ModelPredict");
    seg_modelpredict_ = (Seg_ModelPredict)inferlibrary_->resolve("Seg_ModelPredict");
    cls_modelpredict_ = (Cls_ModelPredict)inferlibrary_->resolve("Cls_ModelPredict");
    mask_modelpredict_ = (Mask_ModelPredict)inferlibrary_->resolve("Mask_ModelPredict");

    // Thread initialization - Configures the reasoning function
    inferthread_ = new InferThread(this);
    inferthread_->setinferfuncs(det_modelpredict_,
                               seg_modelpredict_,
                               cls_modelpredict_,
                               mask_modelpredict_);
    inferthread_->setbtnstop(ui->btnStop);
    inferthread_->setbtninfer(ui->btnInfer);
    inferthread_->setdetthreshold(det_threshold_);
    inferthread_->setinferdelay(infer_delay_);

    // Configure signals and slots
    connect(inferthread_, SIGNAL(InferFinished(QImage*, QImage*)),
            this, SLOT(ImageUpdate(QImage*, QImage*)),
            Qt::BlockingQueuedConnection);
    connect(inferthread_, SIGNAL(SetState_Btn_StopAndInfer(bool , bool )),
            this, SLOT(Btn_StopAndInfer_StateUpdate(bool , bool )),
            Qt::BlockingQueuedConnection);
    connect(inferthread_, SIGNAL(SetCostTime(double )),
            this, SLOT(CostTimeUpdate(double )),
            Qt::BlockingQueuedConnection);
}

void MainWindow::Init_SystemShow()
{
    ui->btnDistory->setEnabled(false);  // There is no initial initialization of the model and the destroy button is invalid
    ui->btnInfer->setEnabled(false);
    ui->btnStop->setEnabled(false);
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    this->Init_SystemState(); // Initialization state at startup
    this->Init_SystemShow();  // Initialize button state at startup
}

MainWindow::~MainWindow()
{
    // Wait for thread to end
    if (inferthread_->doing_infer_)
    {
        inferthread_->break_infer_=false;
        while(inferthread_->doing_infer_==true); // Wait for thread to stop
    }
    // Destruction of the model
    if (has_init_ == true)
    {
        destructmodel_();
    }
    // Unload library
    inferlibrary_->unload();

    delete ui;  // Finally close the screen
}

// Model initialization
void MainWindow::on_btnInit_clicked()
{
    QString dialog_title = "Load model";
    QStringList filters = {"*.pdmodel", "*.pdiparams", "*.yml", "*.yaml"};
    QDir model_dir(QFileDialog::getExistingDirectory(this,
                                                     dialog_title));
    QFileInfoList model_files = model_dir.entryInfoList(filters);  // All valid file names obtained by filtering

    switch (model_files.count()) {
    case 0:
        return;
    case 3:  // det，seg，clas--Load
        for (int i = 0; i < 3; i++)
        {
            QString tag = model_files[i].fileName().split('.')[1];
            if (tag == "pdmodel") model_path_ = model_files[i].filePath();
            else if (tag == "pdiparams") param_path_ = model_files[i].filePath();
            else if (tag == "yml" || tag == "yaml") config_path_ = model_files[i].filePath();
        }
        break;
    case 4: // paddlex和mask--Load
        for (int i = 0; i < 4; i++)
        {
            QString tag = model_files[i].fileName().split('.')[1];
            if (tag == "pdmodel") model_path_ = model_files[i].filePath();
            else if (tag == "pdiparams") param_path_ = model_files[i].filePath();
            else if (tag == "yml" || tag == "yaml")
            {
                if (model_files[i].fileName() == "model.yml" || model_files[i].fileName() == "model.yaml")
                    config_path_ = model_files[i].filePath();
            }
        }
        break;
    default: // Other undefined situations
        QMessageBox::information(this,
            tr("Prompt"),
            tr("Make sure the following files are correctly included in the models folder:\n*.pdmodel, *.pdiparams, *.yml/*.yaml."),
            QMessageBox::Ok,
            QMessageBox::Ok);
        return;
    }

    char paddlex_model_type[10]="";
    try
    {
        if (has_init_ == true)
        {
            destructmodel_();  // Destroy the model and initialize it
            if (model_kind_ == "paddlex")  // Paddlex model special identification
            {
                is_paddlex_ = true;
            }
            if (model_kind_ == "mask")  // MASKRCNN from Paddlex
            {
                model_kind_ = "paddlex";
                is_mask_ = true;
            }

            // input string to char*/[]
            initmodel_(model_kind_.toLocal8Bit().data(),
                      model_path_.toLocal8Bit().data(),
                      param_path_.toLocal8Bit().data(),
                      config_path_.toLocal8Bit().data(),
                      model_env_=="gpu" ? true: false,
                      gpu_id_, paddlex_model_type);

            if (is_paddlex_ && is_mask_==false)  // Replace the actual type of the Paddlex model
            {
                model_kind_ = QString::fromLocal8Bit(paddlex_model_type);  // to real type
                is_paddlex_ = false;
            }
            if (is_paddlex_==false && is_mask_)  // Revert to MASKRCNN type
            {
                model_kind_ = "mask";  // to mask type
                is_mask_ = false;
            }
        }
        else
        {
            if (model_kind_ == "paddlex")
            {
                is_paddlex_ = true;
            }
            if (model_kind_ == "mask")
            {
                model_kind_ = "paddlex";
                is_mask_ = true;
            }

            // input string to char*/[]
            initmodel_(model_kind_.toLocal8Bit().data(),
                      model_path_.toLocal8Bit().data(),
                      param_path_.toLocal8Bit().data(),
                      config_path_.toLocal8Bit().data(),
                      model_env_=="gpu" ? true: false,
                      gpu_id_, paddlex_model_type);

            if (is_paddlex_ && is_mask_==false)
            {
                model_kind_ = QString::fromLocal8Bit(paddlex_model_type);  // to real type
                is_paddlex_ = false;
            }
            if (is_paddlex_==false && is_mask_)
            {
                model_kind_ = "mask";  // to mask type
                is_mask_ = false;
            }

            has_init_ = true; // Initialization is complete
        }
    }
    catch (QException &e) // Failed to initialize a message
    {
        QMessageBox::information(this,
            tr("Initialization failed"),
            tr("1.Please ensure that the model folder correctly contains the following files:\n*.pdmodel, *.pdiparams, *.yml/*.yaml.\n2.Please ensure that the model type is consistent with the loading model."),
            QMessageBox::Ok,
            QMessageBox::Ok);

        if (is_paddlex_)  // Paddlex is not initialized, restore type
        {
            model_kind_ = "paddlex";
            is_paddlex_ = false;
        }
        if (is_mask_)  // Mask is not initialized, restore type
        {
            model_kind_ = "mask";
            is_mask_ = false;
        }

        return;
    }

    inferthread_->setmodeltype(model_kind_); // Set the reasoning interface type
    // Initialization Successful Tips
    QMessageBox::information(this,
        tr("Initialization successful"),
        QString("Model type: ")+model_kind_+QString(", Runtime environment: ")+model_env_+QString("."),
        QMessageBox::Ok,
        QMessageBox::Ok);
    ui->btnInit->setText("模型已初始化");

    ui->btnInfer->setEnabled(true);  // Open reasoning function
    ui->btnDistory->setEnabled(true); // Open the destruction function
    ui->cBoxEnv->setEnabled(false);   // Close the choice of running environment
    ui->cBoxKind->setEnabled(false);   // Turn off the run type selection
    ui->lEditGpuId->setEnabled(false);  // Disable GPU specification
}

// Model destruction
void MainWindow::on_btnDistory_clicked()
{
    if (inferthread_->doing_infer_)  // Reasoning, it cannot be destroyed, issued a hint
    {
        QMessageBox::information(this,
            tr("Prompt"),
            tr("The model is being reasoning, can't be destroyed!\n(after reasoning the completion / termination, the model is destroyed."),
            QMessageBox::Ok,
            QMessageBox::Ok);
        return;
    }

    if (has_init_ == true)
    {
        destructmodel_();
        has_init_ = false;

        QMessageBox::information(this,
            tr("Prompt"),
            tr("The model has been destroyed."),
            QMessageBox::Ok,
            QMessageBox::Ok);
        ui->btnInit->setText("初始化模型");

        ui->btnInfer->setEnabled(false);
        ui->btnDistory->setEnabled(false);
        ui->cBoxEnv->setEnabled(true);
        ui->cBoxKind->setEnabled(true);
        ui->lEditGpuId->setEnabled(true);
    }
    else
    {
        QMessageBox::information(this,
            tr("Prompt"),
            tr("Not initialized, no need to destroy."),
            QMessageBox::Ok,
            QMessageBox::Ok);
    }
}

// Loading pictures
void MainWindow::on_btnLoadImg_clicked()
{
    QString dialog_title = "Loading pictures";
    QString filters = "Loading pictures(*.jpg *.jpeg *.png *.JPEG);;";
    QUrl img_path = QFileDialog::getOpenFileUrl(this,
                                                   dialog_title, QUrl(), filters);
    if (img_path.isEmpty())
    {
        return;
    }

    // if linux or jetson: split--"//"
    // else if windows: split--"///"
    img_file_ = img_path.url().split("//")[1];
    qDebug() << "Input Image Path:" << img_file_ << "\n";

    // Picture reading
    cv::Mat image = cv::imread(img_file_.toLocal8Bit().toStdString());  //BGR
    cv::cvtColor(image, image, COLOR_BGR2RGB);  // BGR --> RGB
    cv::resize(image, image, cv::Size(image.cols/4*4,image.rows/4*4));
    QImage image_from_mat((const uchar*)image.data, image.cols, image.rows, QImage::Format_RGB888);
    QPixmap pixmap(QPixmap::fromImage(image_from_mat));

    // Display images
    ui->labelImage1->setPixmap(pixmap);
    ui->labelImage1->setScaledContents(true);  // Full Label

    inferthread_->setinputimage(img_file_);  // Introduced into the reasoning data
    img_files_ = QStringList();
    video_file_ = "";

    ui->btnLoadImg->setText("图片已加载");
    ui->btnLoadImgs->setText("加载文件夹");
    ui->btnLoadVideo->setText("加载视频");
}

// Loading images folder
void MainWindow::on_btnLoadImgs_clicked()
{
    QString dialog_title = "Loading images folder";
    QStringList filters = {"*.jpg", "*.jpeg", "*.png", "*.JPEG"};
    QDir img_dir(QFileDialog::getExistingDirectory(this,
                                                     dialog_title));
    QFileInfoList img_paths = img_dir.entryInfoList(filters);

    if (img_paths.isEmpty())
    {
        return;
    }

    img_files_.clear();
    for (int i = 0; i < img_paths.count(); i++)
    {
        img_files_.append(img_paths[i].filePath());
    }

    qDebug() << img_files_[0] << "\n";

    // Display the first image
    cv::Mat image = cv::imread(img_files_[0].toLocal8Bit().toStdString());  //BGR
    cv::cvtColor(image, image, COLOR_BGR2RGB);  // BGR --> RGB
    cv::resize(image, image, cv::Size(image.cols/4*4,image.rows/4*4));
    QImage image_from_mat((const uchar*)image.data, image.cols, image.rows, QImage::Format_RGB888);
    QPixmap pixmap(QPixmap::fromImage(image_from_mat));


    ui->labelImage1->setPixmap(pixmap);
    ui->labelImage1->setScaledContents(true);

    inferthread_->setinputimages(img_files_);
    img_file_ = "";
    video_file_ = "";

    ui->btnLoadImg->setText("加载图片");
    ui->btnLoadImgs->setText("文件夹已加载");
    ui->btnLoadVideo->setText("加载视频");
}

// Load the video stream
void MainWindow::on_btnLoadVideo_clicked()
{
    QString dialog_title = "Load video stream";
    QString filters = "video(*.mp4 *.MP4);;";
    QUrl video_path = QFileDialog::getOpenFileUrl(this,
                                                   dialog_title, QUrl(), filters);
    if (video_path.isEmpty())
    {
        return;
    }

    // if linux or jetson: split--"//"
    // else if windows: split--"///"
    video_file_ = video_path.url().split("//")[1];
    qDebug() << "Input Video Path:" << video_file_.toStdString().c_str() << "\n";

    // Display the first image of the video
    VideoCapture capture;
    Mat frame;
    capture.open(video_file_.toLocal8Bit().toStdString()); // Read video
    if(!capture.isOpened())
    {
        QMessageBox::information(this,
            tr("Prompt"),
            tr("1.Video read failed, please check if the video is complete, is it MP4.\n2.(Maybe)Opencv doesn't support the video!"),
            QMessageBox::Ok,
            QMessageBox::Ok);
        return;
    }
    capture >> frame;  // Get the first frame
    cvtColor(frame, frame, COLOR_BGR2RGB);  // BGR --> RGB
    QImage image((const uchar*)frame.data, frame.cols, frame.rows, QImage::Format_RGB888);
    ui->labelImage1->setPixmap(QPixmap::fromImage(image));
    ui->labelImage1->setScaledContents(true);
    capture.release();

    inferthread_->setinputvideo(video_file_);
    img_file_ = "";
    img_files_ = QStringList();

    ui->btnLoadImg->setText("加载图片");
    ui->btnLoadImgs->setText("加载文件夹");
    ui->btnLoadVideo->setText("视频已加载");
}

// Model reasoning - multiple threads
void MainWindow::on_btnInfer_clicked()
{
    if (inferthread_->dataloaded_ != true)  // Data is not loaded
    {
        QMessageBox::information(this,
            tr("Prompt"),
            tr("Please load the data, then reinforce."),
            QMessageBox::Ok,
            QMessageBox::Ok);
        return;
    }
    if (has_init_ == true && inferthread_->doing_infer_ == false)
    {
        ui->btnStop->setEnabled(true);     // Stop button start
        ui->btnInfer->setEnabled(false);     // Inference button off

        // Perform the corresponding type of reasoning
        inferthread_->start();
        QMessageBox::information(this,
            tr("Prompt"),
            tr("Model reasoning"),
            QMessageBox::Ok,
            QMessageBox::Ok);
    }
}

// Reasoning to terminate
void MainWindow::on_btnStop_clicked()
{
    if (inferthread_->doing_infer_ == true)
    {
        QMessageBox::information(this,
            tr("Prompt"),
            tr("Stop model reasoning"),
            QMessageBox::Ok,
            QMessageBox::Ok);

        inferthread_->break_infer_ = true;  // Termination Reasoning -> Auto Go to doing_infer_==false
    }
    else
    {
        QMessageBox::information(this,
            tr("Prompt"),
            tr("Not reasoning, no need to terminate the model reasoning"),
            QMessageBox::Ok,
            QMessageBox::Ok);
    }
}

// Select the model running environment
void MainWindow::on_cBoxEnv_currentIndexChanged(int index)
{
    if (has_init_) // Has been initialized, this operation is invalid
    {
        ui->cBoxEnv->setCurrentIndex(old_model_env_);
        return;
    }
    model_env_ = model_envs_[index];
    old_model_env_ = index;  // Retain this result
}

// Set the model type
void MainWindow::on_cBoxKind_currentIndexChanged(int index)
{
    if (has_init_) // Has been initialized, this operation is invalid
    {
        ui->cBoxKind->setCurrentIndex(old_model_kind_);
        return;
    }
    model_kind_ = model_kinds_[index];
    old_model_kind_ = index;
}

// Set detection threshold
void MainWindow::on_sBoxThreshold_valueChanged(double arg1)
{
    if (inferthread_->doing_infer_)
    {
        ui->sBoxThreshold->setValue(old_det_threshold_);
        return;
    }
    det_threshold_ = (float)arg1;
    inferthread_->setdetthreshold(det_threshold_);

    old_det_threshold_ = det_threshold_;
}

// set gpu_id
void MainWindow::on_lEditGpuId_textChanged(const QString &arg1)
{
    if (has_init_) // Has been initialized, this operation is invalid
    {
        ui->lEditGpuId->setText(QString::number(old_gpu_id_));
        return;
    }
    gpu_id_ = arg1.toInt(); // If you enter a normal number, resolve to the specified number; otherwise 0
    old_gpu_id_ = gpu_id_;
}

// Continuous reasoning interval duration setting
void MainWindow::on_sBoxDelay_valueChanged(const QString &arg1)
{
    if (inferthread_->doing_infer_)
    {
        ui->sBoxDelay->setValue(old_infer_delay_);
        return;
    }
    infer_delay_ = arg1.toInt(); // If you enter a normal number, resolve to the specified number; otherwise 0
    inferthread_->setinferdelay(infer_delay_);

    old_infer_delay_ = infer_delay_;
}


/*  slot funcs blob  */
// update image show
void MainWindow::ImageUpdate(QImage* label1, QImage* label2)
{
    if (inferthread_->doing_infer_)
    {
        ui->labelImage1->clear();
        ui->labelImage1->setPixmap(QPixmap::fromImage(*label1));
        ui->labelImage1->setScaledContents(true); // Full Label

        ui->labelImage2->clear();
        ui->labelImage2->setPixmap(QPixmap::fromImage(*label2));
        ui->labelImage2->setScaledContents(true); // Full Label
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
