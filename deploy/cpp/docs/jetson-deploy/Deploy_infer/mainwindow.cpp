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

    has_Init = false;
    doing_Infer = false;
    is_paddlex = false;
    is_mask = false;

    model_Envs[0] = "cpu";
    model_Envs[1] = "gpu";
    model_Kinds[0] = "det";
    model_Kinds[1] = "seg";
    model_Kinds[2] = "clas";
    model_Kinds[3] = "mask";
    model_Kinds[4] = "paddlex";

    model_Env = "cpu";
    model_Kind = "det";
    gpu_Id = 0;

    det_threshold = 0.5;
    infer_Delay = 50;

    ui->cBoxEnv->setCurrentIndex(0); // Initialize the environment to CPU
    ui->cBoxKind->setCurrentIndex(0); // Initialization type Det
    ui->labelImage1->setStyleSheet("background-color:white;"); // Set the background to initialize the image display area
    ui->labelImage2->setStyleSheet("background-color:white;");

    // Dynamic link library startup
    inferLibrary = new QLibrary("/home/nvidia/Desktop/Deploy_infer/infer_lib/libmodel_infer");
    if(!inferLibrary->load()){
        //Loading failed
        qDebug() << "Load libmodel_infer.so is failed!";
        qDebug() << inferLibrary->errorString();
    }
    else
    {
        qDebug() << "Load libmodel_infer.so is OK!";
    }

    // Export model loading / destruction interface in dynamic library
    initModel = (InitModel)inferLibrary->resolve("InitModel");
    destructModel = (DestructModel)inferLibrary->resolve("DestructModel");
    // Export A model insertion interface in dynamic graph
    det_ModelPredict = (Det_ModelPredict)inferLibrary->resolve("Det_ModelPredict");
    seg_ModelPredict = (Seg_ModelPredict)inferLibrary->resolve("Seg_ModelPredict");
    cls_ModelPredict = (Cls_ModelPredict)inferLibrary->resolve("Cls_ModelPredict");
    mask_ModelPredict = (Mask_ModelPredict)inferLibrary->resolve("Mask_ModelPredict");

    // Thread initialization - Configures the reasoning function
    inferThread = new InferThread(this);
    inferThread->setInferFuncs(det_ModelPredict, seg_ModelPredict, cls_ModelPredict, mask_ModelPredict);
    inferThread->setStopBtn(ui->btnStop);
    inferThread->setInferBtn(ui->btnInfer);
    inferThread->setDetThreshold(det_threshold);
    inferThread->setInferDelay(infer_Delay);

    // Configure signals and slots
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
    if (inferThread->doing_Infer)
    {
        inferThread->break_Infer=false;
        while(inferThread->doing_Infer==true); // Wait for thread to stop
    }
    // Destruction of the model
    if (has_Init == true)
    {
        destructModel();
    }
    // Unload library
    inferLibrary->unload();

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
            if (tag == "pdmodel") model_path = model_files[i].filePath();
            else if (tag == "pdiparams") param_path = model_files[i].filePath();
            else if (tag == "yml" || tag == "yaml") config_path = model_files[i].filePath();
        }
        break;
    case 4: // paddlex和mask--Load
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
        if (has_Init == true)
        {
            destructModel();  // Destroy the model and initialize it
            if (model_Kind == "paddlex")  // Paddlex model special identification
            {
                is_paddlex = true;
            }
            if (model_Kind == "mask")  // MASKRCNN from Paddlex
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

            if (is_paddlex && is_mask==false)  // Replace the actual type of the Paddlex model
            {
                model_Kind = QString::fromLocal8Bit(paddlex_model_type);  // to real type
                is_paddlex = false;
            }
            if (is_paddlex==false && is_mask)  // Revert to MASKRCNN type
            {
                model_Kind = "mask";  // to mask type
                is_mask = false;
            }
        }
        else
        {
            if (model_Kind == "paddlex")
            {
                is_paddlex = true;
            }
            if (model_Kind == "mask")
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

            if (is_paddlex && is_mask==false)
            {
                model_Kind = QString::fromLocal8Bit(paddlex_model_type);  // to real type
                is_paddlex = false;
            }
            if (is_paddlex==false && is_mask)
            {
                model_Kind = "mask";  // to mask type
                is_mask = false;
            }

            has_Init = true; // Initialization is complete
        }
    }
    catch (QException &e) // Failed to initialize a message
    {
        QMessageBox::information(this,
            tr("Initialization failed"),
            tr("1.Please ensure that the model folder correctly contains the following files:\n*.pdmodel, *.pdiparams, *.yml/*.yaml.\n2.Please ensure that the model type is consistent with the loading model."),
            QMessageBox::Ok,
            QMessageBox::Ok);

        if (is_paddlex)  // Paddlex is not initialized, restore type
        {
            model_Kind = "paddlex";
            is_paddlex = false;
        }
        if (is_mask)  // Mask is not initialized, restore type
        {
            model_Kind = "mask";
            is_mask = false;
        }

        return;
    }

    inferThread->setModelType(model_Kind); // Set the reasoning interface type
    // Initialization Successful Tips
    QMessageBox::information(this,
        tr("Initialization successful"),
        QString("Model type: ")+model_Kind+QString(", Runtime environment: ")+model_Env+QString("."),
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
    if (inferThread->doing_Infer)  // Reasoning, it cannot be destroyed, issued a hint
    {
        QMessageBox::information(this,
            tr("Prompt"),
            tr("The model is being reasoning, can't be destroyed!\n(after reasoning the completion / termination, the model is destroyed."),
            QMessageBox::Ok,
            QMessageBox::Ok);
        return;
    }

    if (has_Init == true)
    {
        destructModel();
        has_Init = false;

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

    img_file = img_path.url().split("//")[1];
    qDebug() << "Input Video Path:" << img_file << "\n";

    // Picture reading
    cv::Mat image = cv::imread(img_file.toLocal8Bit().toStdString());  //BGR
    cv::cvtColor(image, image, COLOR_BGR2RGB);  // BGR --> RGB
    cv::resize(image, image, cv::Size(image.cols/4*4,image.rows/4*4));
    QImage image_from_mat((const uchar*)image.data, image.cols, image.rows, QImage::Format_RGB888);
    QPixmap pixmap(QPixmap::fromImage(image_from_mat));

    // Display images
    ui->labelImage1->setPixmap(pixmap);
    ui->labelImage1->setScaledContents(true);  // Full Label

    inferThread->setInputImage(img_file);  // Introduced into the reasoning data
    img_files = QStringList();
    video_file = "";

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

    img_files.clear();
    for (int i = 0; i < img_paths.count(); i++)
    {
        img_files.append(img_paths[i].filePath());
    }

    qDebug() << img_files[0] << "\n";

    // Display the first image
    cv::Mat image = cv::imread(img_files[0].toLocal8Bit().toStdString());  //BGR
    cv::cvtColor(image, image, COLOR_BGR2RGB);  // BGR --> RGB
    cv::resize(image, image, cv::Size(image.cols/4*4,image.rows/4*4));
    QImage image_from_mat((const uchar*)image.data, image.cols, image.rows, QImage::Format_RGB888);
    QPixmap pixmap(QPixmap::fromImage(image_from_mat));


    ui->labelImage1->setPixmap(pixmap);
    ui->labelImage1->setScaledContents(true);

    inferThread->setInputImages(img_files);
    img_file = "";
    video_file = "";

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

    video_file = video_path.url().split("//")[1];
    qDebug() << "Input Video Path:" << video_file.toStdString().c_str() << "\n";

    // Display the first image of the video
    VideoCapture capture;
    Mat frame;
    capture.open(video_file.toLocal8Bit().toStdString()); // Read video
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

    inferThread->setInputVideo(video_file);
    img_file = "";
    img_files = QStringList();

    ui->btnLoadImg->setText("加载图片");
    ui->btnLoadImgs->setText("加载文件夹");
    ui->btnLoadVideo->setText("视频已加载");
}

// Model reasoning - multiple threads
void MainWindow::on_btnInfer_clicked()
{
    if (inferThread->dataLoaded != true)  // Data is not loaded
    {
        QMessageBox::information(this,
            tr("Prompt"),
            tr("Please load the data, then reinforce."),
            QMessageBox::Ok,
            QMessageBox::Ok);
        return;
    }
    if (has_Init == true && inferThread->doing_Infer == false)
    {
        ui->btnStop->setEnabled(true);     // Stop button start
        ui->btnInfer->setEnabled(false);     // Inference button off

        // Perform the corresponding type of reasoning
        inferThread->start();
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
    if (inferThread->doing_Infer == true)
    {
        QMessageBox::information(this,
            tr("Prompt"),
            tr("Stop model reasoning"),
            QMessageBox::Ok,
            QMessageBox::Ok);

        inferThread->break_Infer = true;  // Termination Reasoning -> Auto Go to doing_Infer==false
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
    if (has_Init) // Has been initialized, this operation is invalid
    {
        ui->cBoxEnv->setCurrentIndex(old_model_Env);
        return;
    }
    model_Env = model_Envs[index];
    old_model_Env = index;  // Retain this result
}

// Set the model type
void MainWindow::on_cBoxKind_currentIndexChanged(int index)
{
    if (has_Init) // Has been initialized, this operation is invalid
    {
        ui->cBoxKind->setCurrentIndex(old_model_Kind);
        return;
    }
    model_Kind = model_Kinds[index];
    old_model_Kind = index;
}

// Set detection threshold
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

// set gpu_id
void MainWindow::on_lEditGpuId_textChanged(const QString &arg1)
{
    if (has_Init) // Has been initialized, this operation is invalid
    {
        ui->lEditGpuId->setText(QString::number(old_gpu_Id));
        return;
    }
    gpu_Id = arg1.toInt(); // If you enter a normal number, resolve to the specified number; otherwise 0
    old_gpu_Id = gpu_Id;
}

// Continuous reasoning interval duration setting
void MainWindow::on_sBoxDelay_valueChanged(const QString &arg1)
{
    if (inferThread->doing_Infer)
    {
        ui->sBoxDelay->setValue(old_infer_Delay);
        return;
    }
    infer_Delay = arg1.toInt(); // If you enter a normal number, resolve to the specified number; otherwise 0
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
