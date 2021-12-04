#include "infer_thread.h"
#include <QTimer>
#include <ctime>

// control mainwindow btn
void InferThread::setbtnstop(QPushButton *btn)
{
    btnstop_ = btn;
}

// control mainwindow btn
void InferThread::setbtninfer(QPushButton *btn)
{
    btninfer_ = btn;
}

void InferThread::setdetthreshold(float threshold)
{
    det_threshold_ = threshold;
}

void InferThread::setinferdelay(int delay)
{
    infer_delay_ = delay;
}

void InferThread::get_color_map_list(int num_classes)
{
    uchar *color_list = new uchar[num_classes * 3];
    num_classes += 1;
    for (int i = 1; i < num_classes; i++)
    {
        int j = 0;
        int lab = i;
        while (lab != 0)
        {
            color_list[(i-1) * 3] |= (uchar)(((lab >> 0) & 1)
                                             << (7 - j));
            color_list[(i-1) * 3 + 1] |= (uchar)(((lab >> 1) & 1)
                                                 << (7 - j));
            color_list[(i-1) * 3 + 2] |= (uchar)(((lab >> 2) & 1)
                                                 << (7 - j));

            j += 1;
            lab >>= 3;
        }
    }
    color_map_ = color_list;
}

InferThread::InferThread(QObject *parent) : QThread(parent)
{
    doing_infer_ = false;
    break_infer_ = false;
    dataloaded_ = false;  // false: Unloaded data
    get_color_map_list();

    model_type_ = "det";
    image_path_ = "";
    images_path_ = QStringList();
    video_path_ = "";

    label1_image_ = nullptr;
    label2_image_ = nullptr;

    image1_ = nullptr;
    image2_ = nullptr;
}

void InferThread::setmodeltype(const QString &model_type)
{
    if (model_type=="det")
    {// Check whether the type is met, otherwise set ""
        model_type_ = model_type;
        return;
    }
    else if (model_type=="seg")
    {
        model_type_ = model_type;
        return;
    }
    else if (model_type=="clas")
    {
        model_type_ = model_type;
        return;
    }
    else if (model_type=="mask")
    {
        model_type_ = model_type;
        return;
    }
    else
    {
        // set empty
        model_type_ = "";
    }
}

void InferThread::setinputimage(const QString &image_path)
{
    this->image_path_ = image_path;
    this->images_path_ = QStringList();
    this->video_path_ = "";

    dataloaded_ = true;
}

void InferThread::setinputimages(const QStringList &images_path)
{
    this->images_path_ = images_path;
    this->image_path_ = "";
    this->video_path_ = "";

    dataloaded_ = true;
}

void InferThread::setinputvideo(const QString &video_path)
{
    this->video_path_ = video_path;
    this->image_path_ = "";
    this->images_path_ = QStringList();

    dataloaded_ = true;
}

void InferThread::setinferfuncs(Det_ModelPredict det_inferfunc,
                                Seg_ModelPredict seg_inferfunc,
                                Cls_ModelPredict cls_inferfunc,
                                Mask_ModelPredict mask_inferfunc)
{
    det_modelpredict_ = det_inferfunc;
    seg_modelpredict_ = seg_inferfunc;
    cls_modelpredict_ = cls_inferfunc;
    mask_modelpredict_ = mask_inferfunc;
}

void InferThread::RunInferDet()
{
    if (doing_infer_ == false)
    {
        if (is_inferimage())
        {
            Det_Image();
        }
        else if (is_inferimages())
        {
            Det_Images();
        }
        else if (is_infervideo())
        {
            Det_Video();
        }
    }
}

void InferThread::RunInferSeg()
{
    if (doing_infer_ == false)
    {
        if (is_inferimage())
        {
            Seg_Image();
        }
        else if (is_inferimages())
        {
            Seg_Images();
        }
        else if (is_infervideo())
        {
            Seg_Video();
        }
    }
}

void InferThread::RunInferCls()
{
    if (doing_infer_ == false)
    {
        if (is_inferimage())
        {
            Cls_Image();
        }
        else if (is_inferimages())
        {
            Cls_Images();
        }
        else if (is_infervideo())
        {
            Cls_Video();
        }
    }
}

void InferThread::RunInferMask()
{
    if (doing_infer_ == false)
    {
        if (is_inferimage())
        {
            Mask_Image();
        }
        else if (is_inferimages())
        {
            Mask_Images();
        }
        else if (is_infervideo())
        {
            Mask_Video();
        }
    }
}

// The thread actually runs the configuration
void InferThread::run()
{
    if (model_type_ == "det")
    {
        RunInferDet();
    }
    else if (model_type_ == "seg")
    {
        RunInferSeg();
    }
    else if (model_type_ == "clas")
    {
        RunInferCls();
    }
    else if (model_type_ == "mask")
    {
        RunInferMask();
    }

}

bool InferThread::is_inferimage()
{
    if (image_path_.isEmpty()) return false;
    else return true;
}

bool InferThread::is_inferimages()
{
    if (images_path_.isEmpty()) return false;
    else return true;
}

bool InferThread::is_infervideo()
{
    if (video_path_.isEmpty()) return false;
    else return true;
}

QString InferThread::makelabelinfo(QString label, int id, float score)
{
    QString describe_str = QString::number(id) + ":";
    describe_str += label + "-";
    describe_str += QString::number(score);

    return describe_str;
}

void InferThread::Det_Image()
{
    // Read the picture
    Mat image = imread(image_path_.toLocal8Bit().toStdString());  //BGR

    if (image.cols > 512 || image.rows > 512)
    {
        float ratio = min(image.cols, image.rows) / 512.;
        int new_h = image.cols / ratio;
        int new_w = image.rows / ratio;
        // Make sure pixMap displays properly - cut to scale images
        cv::resize(image, image, cv::Size(new_h/4*4,new_w/4*4));
    }

    // Predict output result
    float bboxs[600];
    int bbox_num[1];
    char labellist[1000];

    // Set the start reasoning state
    doing_infer_ = true;
    try {
        clock_t start_infer_time = clock();
        // Perform reasoning and get results
        qDebug() << "Doing Det-Infer." << "\n";
        det_modelpredict_((const uchar*)image.data, image.cols,
                         image.rows, 3,
                         bboxs, bbox_num, labellist);
        double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
        emit SetCostTime(cost_time);
    } catch (QException &e) {
        // Set the end reasoning state
        doing_infer_ = false;
        qDebug() << "Finished Det-Infer, but it is raise a exception." << "\n";

        emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer
        return;
    }

    // Post-processing
    cvtColor(image, image, COLOR_BGR2RGB);
    if (image1_ == nullptr)
    {
        image1_ = new Mat(image.clone());
    }
    else
    {
        delete image1_;
        image1_ = new Mat(image.clone());
    }
    if (label1_image_ == nullptr)
    {
        label1_image_ = new QImage((const uchar*)image1_->data,
                                   image1_->cols, image1_->rows,
                                   image1_->step, QImage::Format_RGB888);
    }
    else
    {
        delete label1_image_;
        label1_image_ = new QImage((const uchar*)image1_->data,
                                   image1_->cols, image1_->rows,
                                   image1_->step, QImage::Format_RGB888);
    }

    QString labels(labellist);
    QStringList label_list = labels.split(' ');  // Get Label
    for (int i = 0; i < bbox_num[0]; i++)
    {
        int categry_id = (int)bboxs[i*6];
        float score = bboxs[i*6 + 1];
        int left_topx = (int)bboxs[i*6 + 2];
        int left_topy = (int)bboxs[i*6 + 3];
        // Parameters 4 and 5 are width and height,
        // but the same DLL using c# is the lower right vertex
        int right_downx = left_topx + (int)bboxs[i*6 + 4];
        int right_downy = left_topy + (int)bboxs[i*6 + 5];

        if (score >= det_threshold_)
        {
            int color_[3] = { (int)(color_map_[(categry_id % 256) * 3]),
                              (int)(color_map_[(categry_id % 256) * 3 + 1]),
                              (int)(color_map_[(categry_id % 256) * 3 + 2]) };

            QString disscribe_str = makelabelinfo(label_list[i],
                                                  categry_id, score);
            int baseline[1];
            auto text_size = getTextSize(disscribe_str.toStdString(),
                                         FONT_HERSHEY_SIMPLEX,
                              1.0, 2, baseline);
            int text_left_downx = left_topx;
            int text_left_downy = left_topy + text_size.height;

            rectangle(image, Point(left_topx, left_topy),
                      Point(right_downx, right_downy),
                      Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
            putText(image, disscribe_str.toStdString(),
                    Point(text_left_downx, text_left_downy),
                    FONT_HERSHEY_SIMPLEX, 1.0,
                    Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
        }
    }

    if (image2_ == nullptr)
    {
        image2_ = new Mat(image.clone());
    }
    else
    {
        delete image2_;
        image2_ = new Mat(image.clone());
    }
    if (label2_image_ == nullptr)
    {
        label2_image_ = new QImage((const uchar*)image2_->data,
                                   image2_->cols, image2_->rows,
                                   image2_->step, QImage::Format_RGB888);
    }
    else
    {
        delete label2_image_;
        label2_image_ = new QImage((const uchar*)image2_->data,
                                   image2_->cols, image2_->rows,
                                   image2_->step, QImage::Format_RGB888);
    }

    emit InferFinished(label1_image_, label2_image_);

    // Set the end reasoning state
    doing_infer_ = false;
    qDebug() << "Finished Det-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);
}

void InferThread::Det_Images()
{
    doing_infer_ = true;

    for (int j = 0; j < images_path_.count(); j++)
    {
        if (break_infer_) // Exit continuous detection
        {
            doing_infer_ = false;
            break_infer_ = false;

            qDebug() << "Det-Infer has Break." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);
            return;
        }

        QString img_file = images_path_[j];
        Mat image = imread(img_file.toLocal8Bit().toStdString());

        if (image.cols > 512 || image.rows > 512)
        {
            float ratio = min(image.cols, image.rows) / 512.;
            int new_h = image.cols / ratio;
            int new_w = image.rows / ratio;

            cv::resize(image, image, cv::Size(new_h/4*4,new_w/4*4));
        }

        float bboxs[600];
        int bbox_num[1];
        char labellist[1000];

        try {
            clock_t start_infer_time = clock();

            qDebug() << "Doing Det-Infer." << "\n";
            det_modelpredict_((const uchar*)image.data,
                             image.cols, image.rows, 3,
                             bboxs, bbox_num, labellist);
            double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
            emit SetCostTime(cost_time);
        } catch (QException &e) {

            doing_infer_ = false;
            qDebug() << "Finished Det-Infer, but it is raise a exception." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer
            return;
        }

        cvtColor(image, image, COLOR_BGR2RGB);
        if (image1_ == nullptr)
        {
            image1_ = new Mat(image.clone());
        }
        else
        {
            delete image1_;
            image1_ = new Mat(image.clone());
        }
        if (label1_image_ == nullptr)
        {
            label1_image_ = new QImage((const uchar*)image1_->data,
                                       image1_->cols, image1_->rows,
                                       image1_->step, QImage::Format_RGB888);
        }
        else
        {
            delete label1_image_;
            label1_image_ = new QImage((const uchar*)image1_->data,
                                       image1_->cols, image1_->rows,
                                       image1_->step, QImage::Format_RGB888);
        }

        QString labels(labellist);
        QStringList label_list = labels.split(' ');
        for (int i = 0; i < bbox_num[0]; i++)
        {
            int categry_id = (int)bboxs[i*6];
            float score = bboxs[i*6 + 1];
            int left_topx = (int)bboxs[i*6 + 2];
            int left_topy = (int)bboxs[i*6 + 3];
            int right_downx = left_topx + (int)bboxs[i*6 + 4];
            int right_downy = left_topy + (int)bboxs[i*6 + 5];

            if (score >= det_threshold_)
            {
                int color_[3] = { (int)(color_map_[(categry_id % 256) * 3]),
                                  (int)(color_map_[(categry_id % 256) * 3 + 1]),
                                  (int)(color_map_[(categry_id % 256) * 3 + 2]) };

                QString disscribe_str = makelabelinfo(label_list[i],
                                                      categry_id, score);
                int baseline[1];
                auto text_size = getTextSize(disscribe_str.toStdString(),
                                             FONT_HERSHEY_SIMPLEX,
                                  1.0, 2, baseline);
                int text_left_downx = left_topx;
                int text_left_downy = left_topy + text_size.height;

                rectangle(image, Point(left_topx, left_topy),
                          Point(right_downx, right_downy),
                          Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
                putText(image, disscribe_str.toStdString(),
                        Point(text_left_downx, text_left_downy),
                        FONT_HERSHEY_SIMPLEX, 1.0,
                        Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
            }
        }

        if (image2_ == nullptr)
        {
            image2_ = new Mat(image.clone());
        }
        else
        {
            delete image2_;
            image2_ = new Mat(image.clone());
        }
        if (label2_image_ == nullptr)
        {
            label2_image_ = new QImage((const uchar*)image2_->data,
                                       image2_->cols, image2_->rows,
                                       image2_->step, QImage::Format_RGB888);
        }
        else
        {
            delete label2_image_;
            label2_image_ = new QImage((const uchar*)image2_->data,
                                       image2_->cols, image2_->rows,
                                       image2_->step, QImage::Format_RGB888);
        }

        emit InferFinished(label1_image_, label2_image_);

        this->msleep(infer_delay_); // Thread sleep wait
    }

    doing_infer_ = false;
    qDebug() << "Finished Det-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);
}

void InferThread::Det_Video()
{
    doing_infer_ = true;

    VideoCapture cap = VideoCapture(video_path_.toLocal8Bit().toStdString());
    if(!cap.isOpened()) return;

    Mat frame;
    cap >> frame;
    while(!frame.empty()) // Exit the loop if a frame is empty
    {
        if (frame.cols > 512 || frame.rows > 512)
        {
            float ratio = min(frame.cols, frame.rows) / 512.;
            int new_h = frame.cols / ratio;
            int new_w = frame.rows / ratio;

            cv::resize(frame, frame, cv::Size(new_h/4*4,new_w/4*4));
        }

        if (break_infer_)
        {
            doing_infer_ = false;
            break_infer_ = false;

            qDebug() << "Det-Infer has Break." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);
            return;
        }

        float bboxs[600];
        int bbox_num[1];
        char labellist[1000];

        try {
            clock_t start_infer_time = clock();

            qDebug() << "Doing Det-Infer." << "\n";
            det_modelpredict_((const uchar*)frame.data,
                             frame.cols, frame.rows, 3,
                             bboxs, bbox_num, labellist);
            double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
            emit SetCostTime(cost_time);
        } catch (QException &e) {

            doing_infer_ = false;
            qDebug() << "Finished Det-Infer, but it is raise a exception." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);

            return;
        }


        cvtColor(frame, frame, COLOR_BGR2RGB);
        if (image1_ == nullptr)
        {
            image1_ = new Mat(frame.clone());
        }
        else
        {
            delete image1_;
            image1_ = new Mat(frame.clone());
        }
        if (label1_image_ == nullptr)
        {
            label1_image_ = new QImage((const uchar*)image1_->data,
                                       image1_->cols, image1_->rows,
                                       image1_->step, QImage::Format_RGB888);
        }
        else
        {
            delete label1_image_;
            label1_image_ = new QImage((const uchar*)image1_->data,
                                       image1_->cols, image1_->rows,
                                       image1_->step, QImage::Format_RGB888);
        }

        QString labels(labellist);
        QStringList label_list = labels.split(' ');
        for (int i = 0; i < bbox_num[0]; i++)
        {
            int categry_id = (int)bboxs[i*6];
            float score = bboxs[i*6 + 1];
            int left_topx = (int)bboxs[i*6 + 2];
            int left_topy = (int)bboxs[i*6 + 3];
            int right_downx = left_topx + (int)bboxs[i*6 + 4];
            int right_downy = left_topy + (int)bboxs[i*6 + 5];

            if (score >= det_threshold_)
            {
                int color_[3] = { (int)(color_map_[(categry_id % 256) * 3]),
                                  (int)(color_map_[(categry_id % 256) * 3 + 1]),
                                  (int)(color_map_[(categry_id % 256) * 3 + 2]) };

                QString disscribe_str = makelabelinfo(label_list[i],
                                                      categry_id, score);
                int baseline[1];
                auto text_size = getTextSize(disscribe_str.toStdString(),
                                             FONT_HERSHEY_SIMPLEX,
                                  1.0, 2, baseline);
                int text_left_downx = left_topx;
                int text_left_downy = left_topy + text_size.height;

                rectangle(frame, Point(left_topx, left_topy),
                          Point(right_downx, right_downy),
                          Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
                putText(frame, disscribe_str.toStdString(),
                        Point(text_left_downx, text_left_downy),
                        FONT_HERSHEY_SIMPLEX, 1.0,
                        Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
            }
        }

        if (image2_ == nullptr)
        {
            image2_ = new Mat(frame.clone());
        }
        else
        {
            delete image2_;
            image2_ = new Mat(frame.clone());
        }
        if (label2_image_ == nullptr)
        {
            label2_image_ = new QImage((const uchar*)image2_->data,
                                       image2_->cols, image2_->rows,
                                       image2_->step, QImage::Format_RGB888);
        }
        else
        {
            delete label2_image_;
            label2_image_ = new QImage((const uchar*)image2_->data,
                                       image2_->cols, image2_->rows,
                                       image2_->step, QImage::Format_RGB888);
        }

        emit InferFinished(label1_image_, label2_image_);

        cap >> frame;
    }

    doing_infer_ = false;
    qDebug() << "Finished Det-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);
}

void InferThread::Seg_Image()
{
    Mat image = imread(image_path_.toLocal8Bit().toStdString());  //BGR

    if (image.cols > 512 || image.rows > 512)
    {
        float ratio = min(image.cols, image.rows) / 512.;
        int new_h = image.cols / ratio;
        int new_w = image.rows / ratio;

        cv::resize(image, image, cv::Size(new_h/4*4,new_w/4*4));
    }

    // Predict output result
    unsigned char out_image[image.cols * image.rows];

    doing_infer_ = true;
    try {
        clock_t start_infer_time = clock();

        qDebug() << "Doing Seg-Infer." << "\n";
        seg_modelpredict_((const uchar*)image.data,
                         image.cols, image.rows, 3, out_image);
        double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
        emit SetCostTime(cost_time);
    } catch (QException &e) {

        doing_infer_ = false;
        qDebug() << "Finished Seg-Infer, but it is raise a exception." << "\n";

        emit SetState_Btn_StopAndInfer(false, true);

        return;
    }

    // Generate the mask three-channel image
    Mat out3c_image = Mat(image.clone());
    for (int i = 0; i < out3c_image.rows; i++)   // height
    {
        for (int j = 0; j < out3c_image.cols; j++)  // width
        {
            int indexSrc = i*out3c_image.cols + j;

            unsigned char color_id = (int)out_image[indexSrc] % 256; // Pixel category ID

            if (color_id == 0)
                out3c_image.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
            else
                out3c_image.at<Vec3b>(i, j) = Vec3b(color_map_[color_id * 3],
                        color_map_[color_id * 3 + 1], color_map_[color_id * 3 + 2]);
        }
    }

    cvtColor(image, image, COLOR_BGR2RGB);
    if (image1_ == nullptr)
    {
        image1_ = new Mat(image.clone());
    }
    else
    {
        delete image1_;
        image1_ = new Mat(image.clone());
    }
    if (label1_image_ == nullptr)
    {
        label1_image_ = new QImage((const uchar*)image1_->data,
                                   image1_->cols, image1_->rows,
                                   image1_->step, QImage::Format_RGB888);
    }
    else
    {
        delete label1_image_;
        label1_image_ = new QImage((const uchar*)image1_->data,
                                   image1_->cols, image1_->rows,
                                   image1_->step, QImage::Format_RGB888);
    }

    // merge images
    addWeighted(image, 0.5, out3c_image, 0.5, 0, image);

    if (image2_ == nullptr)
    {
        image2_ = new Mat(image.clone());
    }
    else
    {
        delete image2_;
        image2_ = new Mat(image.clone());
    }
    if (label2_image_ == nullptr)
    {
        label2_image_ = new QImage((const uchar*)image2_->data,
                                   image2_->cols, image2_->rows,
                                   image2_->step, QImage::Format_RGB888);
    }
    else
    {
        delete label2_image_;
        label2_image_ = new QImage((const uchar*)image2_->data,
                                   image2_->cols, image2_->rows,
                                   image2_->step, QImage::Format_RGB888);
    }

    emit InferFinished(label1_image_, label2_image_);

    doing_infer_ = false;
    qDebug() << "Finished Seg-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

}

void InferThread::Seg_Images()
{

    doing_infer_ = true;

    for (int j = 0; j < images_path_.count(); j++)
    {
        if (break_infer_)
        {
            doing_infer_ = false;
            break_infer_ = false;

            qDebug() << "Seg-Infer has Break." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);

            return;
        }

        QString img_file = images_path_[j];
        Mat image = imread(img_file.toLocal8Bit().toStdString());

        if (image.cols > 512 || image.rows > 512)
        {
            float ratio = min(image.cols, image.rows) / 512.;
            int new_h = image.cols / ratio;
            int new_w = image.rows / ratio;

            cv::resize(image, image, cv::Size(new_h/4*4,new_w/4*4));
        }

        unsigned char out_image[image.cols * image.rows];
        memset(out_image, 0, sizeof (out_image));

        try {
            clock_t start_infer_time = clock();

            qDebug() << "Doing --Seg Infer." << "\n";
            seg_modelpredict_((const uchar*)image.data,
                             image.cols, image.rows, 3, out_image);
            double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
            emit SetCostTime(cost_time);
        } catch (QException &e) {

            doing_infer_ = false;
            qDebug() << "Finished Seg-Infer, but it is raise a exception." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

            return;
        }


        Mat out3c_image = Mat(image.clone());
        for (int i = 0; i < out3c_image.rows; i++)   // height
        {
            for (int j = 0; j < out3c_image.cols; j++)  // width
            {
                int indexSrc = i*out3c_image.cols + j;

                unsigned char color_id = (int)out_image[indexSrc] % 256;

                if (color_id == 0)
                    out3c_image.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
                else
                    out3c_image.at<Vec3b>(i, j) = Vec3b(color_map_[color_id * 3],
                            color_map_[color_id * 3 + 1], color_map_[color_id * 3 + 2]);
            }
        }


        cvtColor(image, image, COLOR_BGR2RGB);
        if (image1_ == nullptr)
        {
            image1_ = new Mat(image.clone());
        }
        else
        {
            delete image1_;
            image1_ = new Mat(image.clone());
        }
        if (label1_image_ == nullptr)
        {
            label1_image_ = new QImage((const uchar*)image1_->data,
                                       image1_->cols, image1_->rows,
                                       image1_->step, QImage::Format_RGB888);
        }
        else
        {
            delete label1_image_;
            label1_image_ = new QImage((const uchar*)image1_->data,
                                       image1_->cols, image1_->rows,
                                       image1_->step, QImage::Format_RGB888);
        }


        addWeighted(image, 0.5, out3c_image, 0.5, 0, image);

        if (image2_ == nullptr)
        {
            image2_ = new Mat(image.clone());
        }
        else
        {
            delete image2_;
            image2_ = new Mat(image.clone());
        }
        if (label2_image_ == nullptr)
        {
            label2_image_ = new QImage((const uchar*)image2_->data,
                                       image2_->cols, image2_->rows,
                                       image2_->step, QImage::Format_RGB888);
        }
        else
        {
            delete label2_image_;
            label2_image_ = new QImage((const uchar*)image2_->data,
                                       image2_->cols, image2_->rows,
                                       image2_->step, QImage::Format_RGB888);
        }

        emit InferFinished(label1_image_, label2_image_);

        this->msleep(infer_delay_);
    }


    doing_infer_ = false;
    qDebug() << "Finished Seg-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);

}

void InferThread::Seg_Video()
{

    doing_infer_ = true;

    VideoCapture cap = VideoCapture(video_path_.toLocal8Bit().toStdString());
    if(!cap.isOpened()) return;

    Mat frame;
    cap >> frame;
    while(!frame.empty())
    {
        if (frame.cols > 512 || frame.rows > 512)
        {
            float ratio = min(frame.cols, frame.rows) / 512.;
            int new_h = frame.cols / ratio;
            int new_w = frame.rows / ratio;

            cv::resize(frame, frame, cv::Size(new_h/4*4,new_w/4*4));
        }

        if (break_infer_)
        {
            doing_infer_ = false;
            break_infer_ = false;

            qDebug() << "Seg-Infer has Break." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);

            return;
        }

        unsigned char out_image[frame.cols * frame.rows];
        memset(out_image, 0, sizeof (out_image));

        try {
            clock_t start_infer_time = clock();

            qDebug() << "Doing Seg-Infer." << "\n";
            seg_modelpredict_((const uchar*)frame.data,
                             frame.cols, frame.rows, 3, out_image);
            double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
            emit SetCostTime(cost_time);
        } catch (QException &e) {

            doing_infer_ = false;
            qDebug() << "Finished Seg-Infer, but it is raise a exception." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);

            return;
        }


        Mat out3c_image = Mat(frame.clone());
        for (int i = 0; i < out3c_image.rows; i++)   // height
        {
            for (int j = 0; j < out3c_image.cols; j++)  // width
            {
                int indexSrc = i*out3c_image.cols + j;

                unsigned char color_id = (int)out_image[indexSrc] % 256;

                if (color_id == 0)
                    out3c_image.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
                else
                    out3c_image.at<Vec3b>(i, j) = Vec3b(color_map_[color_id * 3],
                            color_map_[color_id * 3 + 1], color_map_[color_id * 3 + 2]);
            }
        }


        cvtColor(frame, frame, COLOR_BGR2RGB);
        if (image1_ == nullptr)
        {
            image1_ = new Mat(frame.clone());
        }
        else
        {
            delete image1_;
            image1_ = new Mat(frame.clone());
        }
        if (label1_image_ == nullptr)
        {
            label1_image_ = new QImage((const uchar*)image1_->data,
                                       image1_->cols, image1_->rows,
                                      image1_->step, QImage::Format_RGB888);
        }
        else
        {
            delete label1_image_;
            label1_image_ = new QImage((const uchar*)image1_->data,
                                       image1_->cols, image1_->rows,
                                      image1_->step, QImage::Format_RGB888);
        }


        addWeighted(frame, 0.5, out3c_image, 0.5, 0, frame);

        if (image2_ == nullptr)
        {
            image2_ = new Mat(frame.clone());
        }
        else
        {
            delete image2_;
            image2_ = new Mat(frame.clone());
        }
        if (label2_image_ == nullptr)
        {
            label2_image_ = new QImage((const uchar*)image2_->data,
                                       image2_->cols, image2_->rows,
                                      image2_->step, QImage::Format_RGB888);
        }
        else
        {
            delete label2_image_;
            label2_image_ = new QImage((const uchar*)image2_->data,
                                       image2_->cols, image2_->rows,
                                      image2_->step, QImage::Format_RGB888);
        }

        emit InferFinished(label1_image_, label2_image_);

        cap >> frame;
    }


    doing_infer_ = false;
    qDebug() << "Finished Seg-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);

}

void InferThread::Cls_Image()
{

    Mat image = imread(image_path_.toLocal8Bit().toStdString());  //BGR

    if (image.cols > 512 || image.rows > 512)
    {
        float ratio = min(image.cols, image.rows) / 512.;
        int new_h = image.cols / ratio;
        int new_w = image.rows / ratio;

        cv::resize(image, image, cv::Size(new_h/4*4,new_w/4*4));
    }

    // Predict output result
    float pre_score[1];
    int pre_category_id[1];
    char pre_category[200];

    doing_infer_ = true;
    try {
        clock_t start_infer_time = clock();

        qDebug() << "Doing Clas-Infer." << "\n";
        cls_modelpredict_((const uchar*)image.data,
                         image.cols, image.rows, 3,
                         pre_score, pre_category, pre_category_id);
        double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
        emit SetCostTime(cost_time);
    } catch (QException &e) {

        doing_infer_ = false;
        qDebug() << "Finished Clas-Infer, but it is raise a exception." << "\n";

        emit SetState_Btn_StopAndInfer(false, true);

        return;
    }


    cvtColor(image, image, COLOR_BGR2RGB);

    float ratio = min(image.cols, image.rows) / 512.;
    int new_h = image.cols / ratio;
    int new_w = image.rows / ratio;
    cv::resize(image, image, cv::Size(new_h/4*4,new_w/4*4));

    if (image1_ == nullptr)
    {
        image1_ = new Mat(image.clone());
    }
    else
    {
        delete image1_;
        image1_ = new Mat(image.clone());
    }
    if (label1_image_ == nullptr)
    {
        label1_image_ = new QImage((const uchar*)image1_->data,
                                   image1_->cols, image1_->rows,
                                  image1_->step, QImage::Format_RGB888);
    }
    else
    {
        delete label1_image_;
        label1_image_ = new QImage((const uchar*)image1_->data,
                                   image1_->cols, image1_->rows,
                                  image1_->step, QImage::Format_RGB888);
    }

    int color_[3] = { (int)(color_map_[(pre_category_id[0] % 256) * 3]),
                      (int)(color_map_[(pre_category_id[0] % 256) * 3 + 1]),
                      (int)(color_map_[(pre_category_id[0] % 256) * 3 + 2]) };

    QString disscribe_str = makelabelinfo(QString(pre_category),
                                          pre_category_id[0], pre_score[0]);
    int baseline[1];
    auto text_size = getTextSize(disscribe_str.toStdString(),
                                 FONT_HERSHEY_SIMPLEX,
                      1.0, 2, baseline);
    int text_left_downx = 0;
    int text_left_downy = 0 + text_size.height;

    putText(image, disscribe_str.toStdString(),
            Point(text_left_downx, text_left_downy),
            FONT_HERSHEY_SIMPLEX, 1.0,
            Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);


    if (image2_ == nullptr)
    {
        image2_ = new Mat(image.clone());
    }
    else
    {
        delete image2_;
        image2_ = new Mat(image.clone());
    }
    if (label2_image_ == nullptr)
    {
        label2_image_ = new QImage((const uchar*)image2_->data,
                                   image2_->cols, image2_->rows,
                                  image2_->step, QImage::Format_RGB888);
    }
    else
    {
        delete label2_image_;
        label2_image_ = new QImage((const uchar*)image2_->data,
                                   image2_->cols, image2_->rows,
                                  image2_->step, QImage::Format_RGB888);
    }

    emit InferFinished(label1_image_, label2_image_);

    doing_infer_ = false;
    qDebug() << "Finished Clas-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);

}

void InferThread::Cls_Images()
{

    doing_infer_ = true;

    for (int j = 0; j < images_path_.count(); j++)
    {
        if (break_infer_)
        {
            doing_infer_ = false;
            break_infer_ = false;

            qDebug() << "Clas-Infer has Break." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);

            return;
        }

        QString img_file = images_path_[j];
        Mat image = imread(img_file.toLocal8Bit().toStdString());

        if (image.cols > 512 || image.rows > 512)
        {
            float ratio = min(image.cols, image.rows) / 512.;
            int new_h = image.cols / ratio;
            int new_w = image.rows / ratio;

            cv::resize(image, image, cv::Size(new_h/4*4,new_w/4*4));
        }


        float pre_score[1];
        int pre_category_id[1];
        char pre_category[200];

        try {
            clock_t start_infer_time = clock();

            qDebug() << "Doing Clas-Infer." << "\n";
            cls_modelpredict_((const uchar*)image.data,
                             image.cols, image.rows, 3,
                             pre_score, pre_category, pre_category_id);
            double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
            emit SetCostTime(cost_time);
        } catch (QException &e) {

            doing_infer_ = false;
            qDebug() << "Finished Clas-Infer, but it is raise a exception." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);

            return;
        }


        cvtColor(image, image, COLOR_BGR2RGB);

        float ratio = min(image.cols, image.rows) / 512.;
        int new_h = image.cols / ratio;
        int new_w = image.rows / ratio;
        cv::resize(image, image, cv::Size(new_h/4*4,new_w/4*4));

        if (image1_ == nullptr)
        {
            image1_ = new Mat(image.clone());
        }
        else
        {
            delete image1_;
            image1_ = new Mat(image.clone());
        }
        if (label1_image_ == nullptr)
        {
            label1_image_ = new QImage((const uchar*)image1_->data,
                                       image1_->cols, image1_->rows,
                                      image1_->step, QImage::Format_RGB888);
        }
        else
        {
            delete label1_image_;
            label1_image_ = new QImage((const uchar*)image1_->data,
                                       image1_->cols, image1_->rows,
                                      image1_->step, QImage::Format_RGB888);
        }

        int color_[3] = { (int)(color_map_[(pre_category_id[0] % 256) * 3]),
                          (int)(color_map_[(pre_category_id[0] % 256) * 3 + 1]),
                          (int)(color_map_[(pre_category_id[0] % 256) * 3 + 2]) };

        QString disscribe_str = makelabelinfo(QString(pre_category),
                                              pre_category_id[0], pre_score[0]);
        int baseline[1];
        auto text_size = getTextSize(disscribe_str.toStdString(),
                                     FONT_HERSHEY_SIMPLEX,
                                     1.0, 2, baseline);
        int text_left_downx = 0;
        int text_left_downy = 0 + text_size.height;

        putText(image, disscribe_str.toStdString(),
                Point(text_left_downx, text_left_downy),
                FONT_HERSHEY_SIMPLEX, 1.0,
                Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);

        if (image2_ == nullptr)
        {
            image2_ = new Mat(image.clone());
        }
        else
        {
            delete image2_;
            image2_ = new Mat(image.clone());
        }
        if (label2_image_ == nullptr)
        {
            label2_image_ = new QImage((const uchar*)image2_->data,
                                       image2_->cols, image2_->rows,
                                      image2_->step, QImage::Format_RGB888);
        }
        else
        {
            delete label2_image_;
            label2_image_ = new QImage((const uchar*)image2_->data,
                                       image2_->cols, image2_->rows,
                                      image2_->step, QImage::Format_RGB888);
        }

        emit InferFinished(label1_image_, label2_image_);

        this->msleep(infer_delay_);
    }


    doing_infer_ = false;
    qDebug() << "Finished Clas-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);

}

void InferThread::Cls_Video()
{

    doing_infer_ = true;

    VideoCapture cap = VideoCapture(video_path_.toLocal8Bit().toStdString());
    if(!cap.isOpened()) return;

    Mat frame;
    cap >> frame;
    while(!frame.empty())
    {
        if (frame.cols > 512 || frame.rows > 512)
        {
            float ratio = min(frame.cols, frame.rows) / 512.;
            int new_h = frame.cols / ratio;
            int new_w = frame.rows / ratio;

            cv::resize(frame, frame, cv::Size(new_h/4*4,new_w/4*4));
        }

        if (break_infer_)
        {
            doing_infer_ = false;
            break_infer_ = false;

            qDebug() << "Clas-Infer has Break." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);

            return;
        }

        float pre_score[1];
        int pre_category_id[1];
        char pre_category[200];

        try {
            clock_t start_infer_time = clock();

            qDebug() << "Doing Clas-Infer." << "\n";
            cls_modelpredict_((const uchar*)frame.data,
                             frame.cols, frame.rows, 3,
                             pre_score, pre_category, pre_category_id);
            double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
            emit SetCostTime(cost_time);
        } catch (QException &e) {

            doing_infer_ = false;
            qDebug() << "Finished Clas-Infer, but it is raise a exception." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);

            return;
        }


        cvtColor(frame, frame, COLOR_BGR2RGB);
        if (image1_ == nullptr)
        {
            image1_ = new Mat(frame.clone());
        }
        else
        {
            delete image1_;
            image1_ = new Mat(frame.clone());
        }
        if (label1_image_ == nullptr)
        {
            label1_image_ = new QImage((const uchar*)image1_->data,
                                       image1_->cols, image1_->rows,
                                      image1_->step, QImage::Format_RGB888);
        }
        else
        {
            delete label1_image_;
            label1_image_ = new QImage((const uchar*)image1_->data,
                                       image1_->cols, image1_->rows,
                                      image1_->step, QImage::Format_RGB888);
        }

        int color_[3] = { (int)(color_map_[(pre_category_id[0] % 256) * 3]),
                          (int)(color_map_[(pre_category_id[0] % 256) * 3 + 1]),
                          (int)(color_map_[(pre_category_id[0] % 256) * 3 + 2]) };

        QString disscribe_str = makelabelinfo(QString(pre_category),
                                              pre_category_id[0], pre_score[0]);
        int baseline[1];
        auto text_size = getTextSize(disscribe_str.toStdString(),
                                     FONT_HERSHEY_SIMPLEX,
                          1.0, 2, baseline);
        int text_left_downx = 0;
        int text_left_downy = 0 + text_size.height;

        putText(frame, disscribe_str.toStdString(),
                Point(text_left_downx, text_left_downy),
                FONT_HERSHEY_SIMPLEX, 1.0,
                Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);

        if (image2_ == nullptr)
        {
            image2_ = new Mat(frame.clone());
        }
        else
        {
            delete image2_;
            image2_ = new Mat(frame.clone());
        }
        if (label2_image_ == nullptr)
        {
            label2_image_ = new QImage((const uchar*)image2_->data,
                                       image2_->cols, image2_->rows,
                                      image2_->step, QImage::Format_RGB888);
        }
        else
        {
            delete label2_image_;
            label2_image_ = new QImage((const uchar*)image2_->data,
                                       image2_->cols, image2_->rows,
                                      image2_->step, QImage::Format_RGB888);
        }

        emit InferFinished(label1_image_, label2_image_);

        cap >> frame;
    }


    doing_infer_ = false;
    qDebug() << "Finished Clas-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);

}

void InferThread::Mask_Image()
{

    Mat image = imread(image_path_.toLocal8Bit().toStdString());  //BGR

    if (image.cols > 512 || image.rows > 512)
    {
        float ratio = min(image.cols, image.rows) / 512.;
        int new_h = image.cols / ratio;
        int new_w = image.rows / ratio;

        cv::resize(image, image, cv::Size(new_h/4*4,new_w/4*4));
    }

    // Predict output result
    float bboxs[600];
    int bbox_num[1];
    char labellist[1000];
    unsigned char out_image[image.cols * image.rows];

    doing_infer_ = true;
    try {
        clock_t start_infer_time = clock();

        qDebug() << "Doing Mask-Infer." << "\n";
        mask_modelpredict_((const uchar*)image.data,
                          image.cols, image.rows, 3,
                          bboxs, out_image, bbox_num, labellist);
        double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
        emit SetCostTime(cost_time);
    } catch (QException &e) {

        doing_infer_ = false;
        qDebug() << "Finished Mask-Infer, but it is raise a exception." << "\n";

        emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

        return;
    }

    Mat out3c_image = Mat(image.clone());
    for (int i = 0; i < out3c_image.rows; i++)   // height
    {
        for (int j = 0; j < out3c_image.cols; j++)  // width
        {
            int indexSrc = i*out3c_image.cols + j;

            unsigned char color_id = (int)out_image[indexSrc] % 256;

            if (color_id == 0)
                out3c_image.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
            else
                out3c_image.at<Vec3b>(i, j) = Vec3b(color_map_[color_id * 3],
                        color_map_[color_id * 3 + 1], color_map_[color_id * 3 + 2]);
        }
    }


    cvtColor(image, image, COLOR_BGR2RGB);
    if (image1_ == nullptr)
    {
        image1_ = new Mat(image.clone());
    }
    else
    {
        delete image1_;
        image1_ = new Mat(image.clone());
    }
    if (label1_image_ == nullptr)
    {
        label1_image_ = new QImage((const uchar*)image1_->data,
                                   image1_->cols, image1_->rows,
                                  image1_->step, QImage::Format_RGB888);
    }
    else
    {
        delete label1_image_;
        label1_image_ = new QImage((const uchar*)image1_->data,
                                   image1_->cols, image1_->rows,
                                  image1_->step, QImage::Format_RGB888);
    }


    addWeighted(image, 0.5, out3c_image, 0.5, 0, image);

    QString labels(labellist);
    QStringList label_list = labels.split(' ');  // Get Label
    for (int i = 0; i < bbox_num[0]; i++)
    {
        int categry_id = (int)bboxs[i*6];
        float score = bboxs[i*6 + 1];
        int left_topx = (int)bboxs[i*6 + 2];
        int left_topy = (int)bboxs[i*6 + 3];
        int right_downx = left_topx + (int)bboxs[i*6 + 4];
        int right_downy = left_topy + (int)bboxs[i*6 + 5];

        if (score >= det_threshold_)
        {
            int color_[3] = { (int)(color_map_[(categry_id % 256) * 3]),
                              (int)(color_map_[(categry_id % 256) * 3 + 1]),
                              (int)(color_map_[(categry_id % 256) * 3 + 2]) };

            QString disscribe_str = makelabelinfo(label_list[i],
                                                  categry_id, score);
            int baseline[1];
            auto text_size = getTextSize(disscribe_str.toStdString(),
                                         FONT_HERSHEY_SIMPLEX,
                              1.0, 2, baseline);
            int text_left_downx = left_topx;
            int text_left_downy = left_topy + text_size.height;

            rectangle(image, Point(left_topx, left_topy),
                      Point(right_downx, right_downy),
                      Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
            putText(image, disscribe_str.toStdString(),
                    Point(text_left_downx, text_left_downy),
                    FONT_HERSHEY_SIMPLEX, 1.0,
                    Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
        }
    }

    if (image2_ == nullptr)
    {
        image2_ = new Mat(image.clone());
    }
    else
    {
        delete image2_;
        image2_ = new Mat(image.clone());
    }
    if (label2_image_ == nullptr)
    {
        label2_image_ = new QImage((const uchar*)image2_->data,
                                   image2_->cols, image2_->rows,
                                  image2_->step, QImage::Format_RGB888);
    }
    else
    {
        delete label2_image_;
        label2_image_ = new QImage((const uchar*)image2_->data,
                                   image2_->cols, image2_->rows,
                                  image2_->step, QImage::Format_RGB888);
    }

    emit InferFinished(label1_image_, label2_image_);


    doing_infer_ = false;
    qDebug() << "Finished Mask-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);

}

void InferThread::Mask_Images()
{

    doing_infer_ = true;

    for (int j = 0; j < images_path_.count(); j++)
    {
        if (break_infer_)
        {
            doing_infer_ = false;
            break_infer_ = false;

            qDebug() << "Mask-Infer has Break." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);

            return;
        }

        QString img_file = images_path_[j];
        Mat image = imread(img_file.toLocal8Bit().toStdString());

        if (image.cols > 512 || image.rows > 512)
        {
            float ratio = min(image.cols, image.rows) / 512.;
            int new_h = image.cols / ratio;
            int new_w = image.rows / ratio;

            cv::resize(image, image, cv::Size(new_h/4*4,new_w/4*4));
        }

        float bboxs[600];
        int bbox_num[1];
        char labellist[1000];
        unsigned char out_image[image.cols * image.rows];
        memset(out_image, 0, sizeof (out_image));

        try {
            clock_t start_infer_time = clock();

            qDebug() << "Doing Mask-Infer." << "\n";
            mask_modelpredict_((const uchar*)image.data,
                              image.cols, image.rows, 3,
                              bboxs, out_image, bbox_num, labellist);
            double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
            emit SetCostTime(cost_time);
        } catch (QException &e) {

            doing_infer_ = false;
            qDebug() << "Finished Mask-Infer, but it is raise a exception." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

            return;
        }


        Mat out3c_image = Mat(image.clone());
        for (int i = 0; i < out3c_image.rows; i++)   // height
        {
            for (int j = 0; j < out3c_image.cols; j++)  // width
            {
                int indexSrc = i*out3c_image.cols + j;

                unsigned char color_id = (int)out_image[indexSrc] % 256;

                if (color_id == 0)
                    out3c_image.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
                else
                    out3c_image.at<Vec3b>(i, j) = Vec3b(color_map_[color_id * 3],
                            color_map_[color_id * 3 + 1], color_map_[color_id * 3 + 2]);
            }
        }


        cvtColor(image, image, COLOR_BGR2RGB);
        if (image1_ == nullptr)
        {
            image1_ = new Mat(image.clone());
        }
        else
        {
            delete image1_;
            image1_ = new Mat(image.clone());
        }
        if (label1_image_ == nullptr)
        {
            label1_image_ = new QImage((const uchar*)image1_->data,
                                       image1_->cols, image1_->rows,
                                      image1_->step, QImage::Format_RGB888);
        }
        else
        {
            delete label1_image_;
            label1_image_ = new QImage((const uchar*)image1_->data,
                                       image1_->cols, image1_->rows,
                                      image1_->step, QImage::Format_RGB888);
        }

        addWeighted(image, 0.5, out3c_image, 0.5, 0, image);

        QString labels(labellist);
        QStringList label_list = labels.split(' ');  // 获取label
        for (int i = 0; i < bbox_num[0]; i++)
        {
            int categry_id = (int)bboxs[i*6];
            float score = bboxs[i*6 + 1];
            int left_topx = (int)bboxs[i*6 + 2];
            int left_topy = (int)bboxs[i*6 + 3];
            int right_downx = left_topx + (int)bboxs[i*6 + 4];
            int right_downy = left_topy + (int)bboxs[i*6 + 5];

            if (score >= det_threshold_)
            {
                int color_[3] = { (int)(color_map_[(categry_id % 256) * 3]),
                                  (int)(color_map_[(categry_id % 256) * 3 + 1]),
                                  (int)(color_map_[(categry_id % 256) * 3 + 2]) };

                QString disscribe_str = makelabelinfo(label_list[i],
                                                      categry_id, score);
                int baseline[1];
                auto text_size = getTextSize(disscribe_str.toStdString(),
                                             FONT_HERSHEY_SIMPLEX,
                                  1.0, 2, baseline);
                int text_left_downx = left_topx;
                int text_left_downy = left_topy + text_size.height;

                rectangle(image, Point(left_topx, left_topy),
                          Point(right_downx, right_downy),
                          Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
                putText(image, disscribe_str.toStdString(),
                        Point(text_left_downx, text_left_downy),
                        FONT_HERSHEY_SIMPLEX, 1.0,
                        Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
            }
        }

        if (image2_ == nullptr)
        {
            image2_ = new Mat(image.clone());
        }
        else
        {
            delete image2_;
            image2_ = new Mat(image.clone());
        }
        if (label2_image_ == nullptr)
        {
            label2_image_ = new QImage((const uchar*)image2_->data,
                                       image2_->cols, image2_->rows,
                                      image2_->step, QImage::Format_RGB888);
        }
        else
        {
            delete label2_image_;
            label2_image_ = new QImage((const uchar*)image2_->data,
                                       image2_->cols, image2_->rows,
                                      image2_->step, QImage::Format_RGB888);
        }

        emit InferFinished(label1_image_, label2_image_);

        this->msleep(infer_delay_);
    }

    doing_infer_ = false;
    qDebug() << "Finished Mask-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);

}

void InferThread::Mask_Video()
{
    doing_infer_ = true;

    VideoCapture cap = VideoCapture(video_path_.toLocal8Bit().toStdString());
    if(!cap.isOpened()) return;

    Mat frame;
    cap >> frame;
    while(!frame.empty())
    {
        if (frame.cols > 512 || frame.rows > 512)
        {
            float ratio = min(frame.cols, frame.rows) / 512.;
            int new_h = frame.cols / ratio;
            int new_w = frame.rows / ratio;

            cv::resize(frame, frame, cv::Size(new_h/4*4,new_w/4*4));
        }

        if (break_infer_)
        {
            doing_infer_ = false;
            break_infer_ = false;

            qDebug() << "Mask-Infer has Break." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);

            return;
        }


        float bboxs[600];
        int bbox_num[1];
        char labellist[1000];
        unsigned char out_image[frame.cols * frame.rows];
        memset(out_image, 0, sizeof (out_image));

        try {
            clock_t start_infer_time = clock();

            qDebug() << "Doing Mask-Infer." << "\n";
            mask_modelpredict_((const uchar*)frame.data,
                              frame.cols, frame.rows, 3, bboxs,
                              out_image, bbox_num, labellist);
            double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
            emit SetCostTime(cost_time);
        } catch (QException &e) {

            doing_infer_ = false;
            qDebug() << "Finished Mask-Infer, but it is raise a exception." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

            return;
        }

        Mat out3c_image = Mat(frame.clone());
        for (int i = 0; i < out3c_image.rows; i++)   // height
        {
            for (int j = 0; j < out3c_image.cols; j++)  // width
            {
                int indexSrc = i*out3c_image.cols + j;

                unsigned char color_id = (int)out_image[indexSrc] % 256;

                if (color_id == 0)
                    out3c_image.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
                else
                    out3c_image.at<Vec3b>(i, j) = Vec3b(color_map_[color_id * 3],
                            color_map_[color_id * 3 + 1], color_map_[color_id * 3 + 2]);
            }
        }

        cvtColor(frame, frame, COLOR_BGR2RGB);
        if (image1_ == nullptr)
        {
            image1_ = new Mat(frame.clone());
        }
        else
        {
            delete image1_;
            image1_ = new Mat(frame.clone());
        }
        if (label1_image_ == nullptr)
        {
            label1_image_ = new QImage((const uchar*)image1_->data,
                                       image1_->cols, image1_->rows,
                                      image1_->step, QImage::Format_RGB888);
        }
        else
        {
            delete label1_image_;
            label1_image_ = new QImage((const uchar*)image1_->data,
                                       image1_->cols, image1_->rows,
                                      image1_->step, QImage::Format_RGB888);
        }

        addWeighted(frame, 0.5, out3c_image, 0.5, 0, frame);

        QString labels(labellist);
        QStringList label_list = labels.split(' ');
        for (int i = 0; i < bbox_num[0]; i++)
        {
            int categry_id = (int)bboxs[i*6];
            float score = bboxs[i*6 + 1];
            int left_topx = (int)bboxs[i*6 + 2];
            int left_topy = (int)bboxs[i*6 + 3];
            int right_downx = left_topx + (int)bboxs[i*6 + 4];
            int right_downy = left_topy + (int)bboxs[i*6 + 5];

            if (score >= det_threshold_)
            {
                int color_[3] = { (int)(color_map_[(categry_id % 256) * 3]),
                                  (int)(color_map_[(categry_id % 256) * 3 + 1]),
                                  (int)(color_map_[(categry_id % 256) * 3 + 2]) };

                QString disscribe_str = makelabelinfo(label_list[i],
                                                      categry_id, score);
                int baseline[1];
                auto text_size = getTextSize(disscribe_str.toStdString(),
                                             FONT_HERSHEY_SIMPLEX,
                                  1.0, 2, baseline);
                int text_left_downx = left_topx;
                int text_left_downy = left_topy + text_size.height;

                rectangle(frame, Point(left_topx, left_topy),
                          Point(right_downx, right_downy),
                          Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
                putText(frame, disscribe_str.toStdString(),
                        Point(text_left_downx, text_left_downy),
                        FONT_HERSHEY_SIMPLEX, 1.0,
                        Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
            }
        }

        if (image2_ == nullptr)
        {
            image2_ = new Mat(frame.clone());
        }
        else
        {
            delete image2_;
            image2_ = new Mat(frame.clone());
        }
        if (label2_image_ == nullptr)
        {
            label2_image_ = new QImage((const uchar*)image2_->data,
                                       image2_->cols, image2_->rows,
                                      image2_->step, QImage::Format_RGB888);
        }
        else
        {
            delete label2_image_;
            label2_image_ = new QImage((const uchar*)image2_->data,
                                       image2_->cols, image2_->rows,
                                      image2_->step, QImage::Format_RGB888);
        }

        emit InferFinished(label1_image_, label2_image_);

        cap >> frame;
    }

    doing_infer_ = false;
    qDebug() << "Finished Mask-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);

}
