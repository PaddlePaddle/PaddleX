#include "inferthread.h"
#include <QTimer>
#include <ctime>

void InferThread::setStopBtn(QPushButton *btn)
{
    btnStop = btn;
}

void InferThread::setInferBtn(QPushButton *btn)
{
    btnInfer = btn;
}

void InferThread::setDetThreshold(float threshold)
{
    det_Threshold = threshold;
}

void InferThread::setInferDelay(int delay)
{
    infer_Delay = delay;
}

uchar *InferThread::get_color_map_list(int num_classes)
{
    uchar *color_list = new uchar[num_classes * 3];
    num_classes += 1;
    for (int i = 1; i < num_classes; i++)
    {
        int j = 0;
        int lab = i;
        while (lab != 0)
        {
            color_list[(i-1) * 3] |= (uchar)(((lab >> 0) & 1) << (7 - j));
            color_list[(i-1) * 3 + 1] |= (uchar)(((lab >> 1) & 1) << (7 - j));
            color_list[(i-1) * 3 + 2] |= (uchar)(((lab >> 2) & 1) << (7 - j));

            j += 1;
            lab >>= 3;
        }
    }
    return color_list;
}

InferThread::InferThread(QObject *parent) : QThread(parent)
{
    doing_Infer = false;
    break_Infer = false;
    dataLoaded = false;  // false: Unloaded data
    color_map = get_color_map_list();

    model_Type = "det";
    image_path = "";
    images_path = QStringList();
    video_path = "";

    label1_image = nullptr;
    label2_image = nullptr;

    image1 = nullptr;
    image2 = nullptr;
}

void InferThread::setModelType(QString &model_type)
{
    if (model_type=="det") // Check whether the type is met, otherwise set ""
    {
        model_Type = model_type;
        return;
    }
    else if (model_type=="seg")
    {
        model_Type = model_type;
        return;
    }
    else if (model_type=="clas")
    {
        model_Type = model_type;
        return;
    }
    else if (model_type=="mask")
    {
        model_Type = model_type;
        return;
    }
    else
    {
        // set empty
        model_Type = "";
    }
}

void InferThread::setInputImage(QString &image_path)
{
    this->image_path = image_path;
    this->images_path = QStringList();
    this->video_path = "";

    dataLoaded = true;
}

void InferThread::setInputImages(QStringList &images_path)
{
    this->images_path = images_path;
    this->image_path = "";
    this->video_path = "";

    dataLoaded = true;
}

void InferThread::setInputVideo(QString &video_path)
{
    this->video_path = video_path;
    this->image_path = "";
    this->images_path = QStringList();

    dataLoaded = true;
}

void InferThread::setInferFuncs(Det_ModelPredict det_Inferfunc, Seg_ModelPredict seg_Inferfunc, Cls_ModelPredict cls_Inferfunc, Mask_ModelPredict mask_Inferfunc)
{
    det_ModelPredict = det_Inferfunc;
    seg_ModelPredict = seg_Inferfunc;
    cls_ModelPredict = cls_Inferfunc;
    mask_ModelPredict = mask_Inferfunc;
}

void InferThread::runInferDet()
{
    if (doing_Infer == false)
    {
        if (is_InferImage())
        {
            Det_Image();
        }
        else if (is_InferImages())
        {
            Det_Images();
        }
        else if (is_InferVideo())
        {
            Det_Video();
        }
    }
    else
    {
        // TODO
    }
}

void InferThread::runInferSeg()
{
    if (doing_Infer == false)
    {
        if (is_InferImage())
        {
            Seg_Image();
        }
        else if (is_InferImages())
        {
            Seg_Images();
        }
        else if (is_InferVideo())
        {
            Seg_Video();
        }
    }
    else
    {
        // TODO
    }
}

void InferThread::runInferCls()
{
    if (doing_Infer == false)
    {
        if (is_InferImage())
        {
            Cls_Image();
        }
        else if (is_InferImages())
        {
            Cls_Images();
        }
        else if (is_InferVideo())
        {
            Cls_Video();
        }
    }
    else
    {
        // TODO
    }
}

void InferThread::runInferMask()
{
    if (doing_Infer == false)
    {
        if (is_InferImage())
        {
            Mask_Image();
        }
        else if (is_InferImages())
        {
            Mask_Images();
        }
        else if (is_InferVideo())
        {
            Mask_Video();
        }
    }
    else
    {
        // TODO
    }
}

// The thread actually runs the configuration
void InferThread::run()
{
    if (model_Type == "det")
    {
        runInferDet();
    }
    else if (model_Type == "seg")
    {
        runInferSeg();
    }
    else if (model_Type == "clas")
    {
        runInferCls();
    }
    else if (model_Type == "mask")
    {
        runInferMask();
    }

}

bool InferThread::is_InferImage()
{
    if (image_path.isEmpty()) return false;
    else return true;
}

bool InferThread::is_InferImages()
{
    if (images_path.isEmpty()) return false;
    else return true;
}

bool InferThread::is_InferVideo()
{
    if (video_path.isEmpty()) return false;
    else return true;
}

QString InferThread::makeLabelInfo(QString label, int id, float score)
{
    QString describe_str = QString::number(id) + ":";
    describe_str += label + "-";
    describe_str += QString::number(score);

    return describe_str;
}

void InferThread::Det_Image()
{
    // Read the picture
    Mat image = imread(image_path.toLocal8Bit().toStdString());  //BGR

    if (image.cols > 512 || image.rows > 512)
    {
        float ratio = min(image.cols, image.rows) / 512.;
        int new_h = image.cols / ratio;
        int new_w = image.rows / ratio;

        cv::resize(image, image, cv::Size(new_h/4*4,new_w/4*4)); // Make sure pixMap displays properly - cut to scale images
    }

    // Predict output result
    float bboxs[600];
    int bbox_num[1];
    char labellist[1000];

    // Set the start reasoning state
    doing_Infer = true;
    try {
        clock_t start_infer_time = clock();
        // Perform reasoning and get results
        qDebug() << "Doing Det-Infer." << "\n";
        det_ModelPredict((const uchar*)image.data, image.cols, image.rows, 3, bboxs, bbox_num, labellist);
        double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
        emit SetCostTime(cost_time);
    } catch (QException &e) {
        // Set the end reasoning state
        doing_Infer = false;
        qDebug() << "Finished Det-Infer, but it is raise a exception." << "\n";

        emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer
//    btnStop->setEnabled(false);  // When reasoning is complete, close the button of reasoning interruption to prevent late point
//    btnInfer->setEnabled(true);  // When the reasoning is complete, the button for the execution of the reasoning is opened to allow the reasoning again
        return;
    }

    // Post-processing
    cvtColor(image, image, COLOR_BGR2RGB);
    if (image1 == nullptr)
    {
        image1 = new Mat(image.clone());
    }
    else
    {
        delete image1;
        image1 = new Mat(image.clone());
    }
    if (label1_image == nullptr)
    {
        label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                  image1->step, QImage::Format_RGB888);
    }
    else
    {
        delete label1_image;
        label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                  image1->step, QImage::Format_RGB888);
    }

    QString labels(labellist);
    QStringList label_list = labels.split(' ');  // Get Label
    for (int i = 0; i < bbox_num[0]; i++)
    {
        int categry_id = (int)bboxs[i*6];
        float score = bboxs[i*6 + 1];
        int left_topx = (int)bboxs[i*6 + 2];
        int left_topy = (int)bboxs[i*6 + 3];
        int right_downx = left_topx + (int)bboxs[i*6 + 4];  // Parameters 4 and 5 are width and height, but the same DLL using c# is the lower right vertex
        int right_downy = left_topy + (int)bboxs[i*6 + 5];

        if (score >= det_Threshold)
        {
            int color_[3] = { (int)(color_map[(categry_id % 256) * 3]),
                              (int)(color_map[(categry_id % 256) * 3 + 1]),
                              (int)(color_map[(categry_id % 256) * 3 + 2]) };

            QString disscribe_str = makeLabelInfo(label_list[i], categry_id, score);
            int baseline[1];
            auto text_size = getTextSize(disscribe_str.toStdString(), FONT_HERSHEY_SIMPLEX,
                              1.0, 2, baseline);
            int text_left_downx = left_topx; // Small offset adjustment: (int)(text_size.Width/10)
            int text_left_downy = left_topy + text_size.height;

            rectangle(image, Point(left_topx, left_topy),
                      Point(right_downx, right_downy),
                      Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
            putText(image, disscribe_str.toStdString(), Point(text_left_downx, text_left_downy),
                    FONT_HERSHEY_SIMPLEX, 1.0,
                    Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
        }
    }

    if (image2 == nullptr)
    {
        image2 = new Mat(image.clone());
    }
    else
    {
        delete image2;
        image2 = new Mat(image.clone());
    }
    if (label2_image == nullptr)
    {
        label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                  image2->step, QImage::Format_RGB888);
    }
    else
    {
        delete label2_image;
        label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                  image2->step, QImage::Format_RGB888);
    }

    emit InferFinished(label1_image, label2_image);

    // Set the end reasoning state
    doing_Infer = false;
    qDebug() << "Finished Det-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer
//    btnStop->setEnabled(false);
//    btnInfer->setEnabled(true);
}

void InferThread::Det_Images()
{
    doing_Infer = true;

    for (int j = 0; j < images_path.count(); j++)
    {
        if (break_Infer) // Exit continuous detection
        {
            doing_Infer = false;
            break_Infer = false;

            qDebug() << "Det-Infer has Break." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer
            return;
        }

        QString img_file = images_path[j]; // Get image path
        Mat image = imread(img_file.toLocal8Bit().toStdString());  // Help with Chinese paths

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
            det_ModelPredict((const uchar*)image.data, image.cols, image.rows, 3, bboxs, bbox_num, labellist);
            double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
            emit SetCostTime(cost_time);
        } catch (QException &e) {

            doing_Infer = false;
            qDebug() << "Finished Det-Infer, but it is raise a exception." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer
            return;
        }

        cvtColor(image, image, COLOR_BGR2RGB);
        if (image1 == nullptr)
        {
            image1 = new Mat(image.clone());
        }
        else
        {
            delete image1;
            image1 = new Mat(image.clone());
        }
        if (label1_image == nullptr)
        {
            label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                      image1->step, QImage::Format_RGB888);
        }
        else
        {
            delete label1_image;
            label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                      image1->step, QImage::Format_RGB888);
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

            if (score >= det_Threshold)
            {
                int color_[3] = { (int)(color_map[(categry_id % 256) * 3]),
                                  (int)(color_map[(categry_id % 256) * 3 + 1]),
                                  (int)(color_map[(categry_id % 256) * 3 + 2]) };

                QString disscribe_str = makeLabelInfo(label_list[i], categry_id, score);
                int baseline[1];
                auto text_size = getTextSize(disscribe_str.toStdString(), FONT_HERSHEY_SIMPLEX,
                                  1.0, 2, baseline);
                int text_left_downx = left_topx;
                int text_left_downy = left_topy + text_size.height;

                rectangle(image, Point(left_topx, left_topy),
                          Point(right_downx, right_downy),
                          Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
                putText(image, disscribe_str.toStdString(), Point(text_left_downx, text_left_downy),
                        FONT_HERSHEY_SIMPLEX, 1.0,
                        Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
            }
        }

        if (image2 == nullptr)
        {
            image2 = new Mat(image.clone());
        }
        else
        {
            delete image2;
            image2 = new Mat(image.clone());
        }
        if (label2_image == nullptr)
        {
            label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                      image2->step, QImage::Format_RGB888);
        }
        else
        {
            delete label2_image;
            label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                      image2->step, QImage::Format_RGB888);
        }

        emit InferFinished(label1_image, label2_image);

        this->msleep(infer_Delay); // Thread sleep wait
    }

    doing_Infer = false;
    qDebug() << "Finished Det-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer
}

void InferThread::Det_Video()
{
    doing_Infer = true;

    VideoCapture cap = VideoCapture(video_path.toLocal8Bit().toStdString());
    if(!cap.isOpened()) return; // Return if the video does not open properly

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

        if (break_Infer)
        {
            doing_Infer = false;
            break_Infer = false;

            qDebug() << "Det-Infer has Break." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer
            return;
        }

        float bboxs[600];
        int bbox_num[1];
        char labellist[1000];

        try {
            clock_t start_infer_time = clock();

            qDebug() << "Doing Det-Infer." << "\n";
            det_ModelPredict((const uchar*)frame.data, frame.cols, frame.rows, 3, bboxs, bbox_num, labellist);
            double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
            emit SetCostTime(cost_time);
        } catch (QException &e) {

            doing_Infer = false;
            qDebug() << "Finished Det-Infer, but it is raise a exception." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

            return;
        }


        cvtColor(frame, frame, COLOR_BGR2RGB);
        if (image1 == nullptr)
        {
            image1 = new Mat(frame.clone());
        }
        else
        {
            delete image1;
            image1 = new Mat(frame.clone());
        }
        if (label1_image == nullptr)
        {
            label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                      image1->step, QImage::Format_RGB888);
        }
        else
        {
            delete label1_image;
            label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                      image1->step, QImage::Format_RGB888);
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

            if (score >= det_Threshold)
            {
                int color_[3] = { (int)(color_map[(categry_id % 256) * 3]),
                                  (int)(color_map[(categry_id % 256) * 3 + 1]),
                                  (int)(color_map[(categry_id % 256) * 3 + 2]) };

                QString disscribe_str = makeLabelInfo(label_list[i], categry_id, score);
                int baseline[1];
                auto text_size = getTextSize(disscribe_str.toStdString(), FONT_HERSHEY_SIMPLEX,
                                  1.0, 2, baseline);
                int text_left_downx = left_topx;
                int text_left_downy = left_topy + text_size.height;

                rectangle(frame, Point(left_topx, left_topy),
                          Point(right_downx, right_downy),
                          Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
                putText(frame, disscribe_str.toStdString(), Point(text_left_downx, text_left_downy),
                        FONT_HERSHEY_SIMPLEX, 1.0,
                        Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
            }
        }

        if (image2 == nullptr)
        {
            image2 = new Mat(frame.clone());
        }
        else
        {
            delete image2;
            image2 = new Mat(frame.clone());
        }
        if (label2_image == nullptr)
        {
            label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                      image2->step, QImage::Format_RGB888);
        }
        else
        {
            delete label2_image;
            label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                      image2->step, QImage::Format_RGB888);
        }

        emit InferFinished(label1_image, label2_image);

        cap >> frame;
    }

    doing_Infer = false;
    qDebug() << "Finished Det-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer
}

void InferThread::Seg_Image()
{
    Mat image = imread(image_path.toLocal8Bit().toStdString());  //BGR

    if (image.cols > 512 || image.rows > 512)
    {
        float ratio = min(image.cols, image.rows) / 512.;
        int new_h = image.cols / ratio;
        int new_w = image.rows / ratio;

        cv::resize(image, image, cv::Size(new_h/4*4,new_w/4*4));
    }

    // Predict output result
    unsigned char out_image[image.cols * image.rows];

    doing_Infer = true;
    try {
        clock_t start_infer_time = clock();

        qDebug() << "Doing Seg-Infer." << "\n";
        seg_ModelPredict((const uchar*)image.data, image.cols, image.rows, 3, out_image);
        double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
        emit SetCostTime(cost_time);
    } catch (QException &e) {

        doing_Infer = false;
        qDebug() << "Finished Seg-Infer, but it is raise a exception." << "\n";

        emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

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
                out3c_image.at<Vec3b>(i, j) = Vec3b(color_map[color_id * 3], color_map[color_id * 3 + 1], color_map[color_id * 3 + 2]);
        }
    }

    cvtColor(image, image, COLOR_BGR2RGB);
    if (image1 == nullptr)
    {
        image1 = new Mat(image.clone());
    }
    else
    {
        delete image1;
        image1 = new Mat(image.clone());
    }
    if (label1_image == nullptr)
    {
        label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                  image1->step, QImage::Format_RGB888);
    }
    else
    {
        delete label1_image;
        label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                  image1->step, QImage::Format_RGB888);
    }

    // merge images
    addWeighted(image, 0.5, out3c_image, 0.5, 0, image);

    if (image2 == nullptr)
    {
        image2 = new Mat(image.clone());
    }
    else
    {
        delete image2;
        image2 = new Mat(image.clone());
    }
    if (label2_image == nullptr)
    {
        label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                  image2->step, QImage::Format_RGB888);
    }
    else
    {
        delete label2_image;
        label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                  image2->step, QImage::Format_RGB888);
    }

    emit InferFinished(label1_image, label2_image);

    doing_Infer = false;
    qDebug() << "Finished Seg-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

}

void InferThread::Seg_Images()
{

    doing_Infer = true;

    for (int j = 0; j < images_path.count(); j++)
    {
        if (break_Infer)
        {
            doing_Infer = false;
            break_Infer = false;

            qDebug() << "Seg-Infer has Break." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

            return;
        }

        QString img_file = images_path[j];
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
            seg_ModelPredict((const uchar*)image.data, image.cols, image.rows, 3, out_image);
            double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
            emit SetCostTime(cost_time);
        } catch (QException &e) {

            doing_Infer = false;
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
                    out3c_image.at<Vec3b>(i, j) = Vec3b(color_map[color_id * 3], color_map[color_id * 3 + 1], color_map[color_id * 3 + 2]);
            }
        }


        cvtColor(image, image, COLOR_BGR2RGB);
        if (image1 == nullptr)
        {
            image1 = new Mat(image.clone());
        }
        else
        {
            delete image1;
            image1 = new Mat(image.clone());
        }
        if (label1_image == nullptr)
        {
            label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                      image1->step, QImage::Format_RGB888);
        }
        else
        {
            delete label1_image;
            label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                      image1->step, QImage::Format_RGB888);
        }


        addWeighted(image, 0.5, out3c_image, 0.5, 0, image);

        if (image2 == nullptr)
        {
            image2 = new Mat(image.clone());
        }
        else
        {
            delete image2;
            image2 = new Mat(image.clone());
        }
        if (label2_image == nullptr)
        {
            label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                      image2->step, QImage::Format_RGB888);
        }
        else
        {
            delete label2_image;
            label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                      image2->step, QImage::Format_RGB888);
        }

        emit InferFinished(label1_image, label2_image);

        this->msleep(infer_Delay);
    }


    doing_Infer = false;
    qDebug() << "Finished Seg-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

}

void InferThread::Seg_Video()
{

    doing_Infer = true;

    VideoCapture cap = VideoCapture(video_path.toLocal8Bit().toStdString());
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

        if (break_Infer)
        {
            doing_Infer = false;
            break_Infer = false;

            qDebug() << "Seg-Infer has Break." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

            return;
        }

        unsigned char out_image[frame.cols * frame.rows];
        memset(out_image, 0, sizeof (out_image));

        try {
            clock_t start_infer_time = clock();
            
            qDebug() << "Doing Seg-Infer." << "\n";
            seg_ModelPredict((const uchar*)frame.data, frame.cols, frame.rows, 3, out_image);
            double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
            emit SetCostTime(cost_time);
        } catch (QException &e) {

            doing_Infer = false;
            qDebug() << "Finished Seg-Infer, but it is raise a exception." << "\n";

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
                    out3c_image.at<Vec3b>(i, j) = Vec3b(color_map[color_id * 3], color_map[color_id * 3 + 1], color_map[color_id * 3 + 2]);
            }
        }


        cvtColor(frame, frame, COLOR_BGR2RGB);
        if (image1 == nullptr)
        {
            image1 = new Mat(frame.clone());
        }
        else
        {
            delete image1;
            image1 = new Mat(frame.clone());
        }
        if (label1_image == nullptr)
        {
            label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                      image1->step, QImage::Format_RGB888);
        }
        else
        {
            delete label1_image;
            label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                      image1->step, QImage::Format_RGB888);
        }


        addWeighted(frame, 0.5, out3c_image, 0.5, 0, frame);

        if (image2 == nullptr)
        {
            image2 = new Mat(frame.clone());
        }
        else
        {
            delete image2;
            image2 = new Mat(frame.clone());
        }
        if (label2_image == nullptr)
        {
            label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                      image2->step, QImage::Format_RGB888);
        }
        else
        {
            delete label2_image;
            label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                      image2->step, QImage::Format_RGB888);
        }

        emit InferFinished(label1_image, label2_image);

        cap >> frame;
    }


    doing_Infer = false;
    qDebug() << "Finished Seg-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

}

void InferThread::Cls_Image()
{

    Mat image = imread(image_path.toLocal8Bit().toStdString());  //BGR

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

    doing_Infer = true;
    try {
        clock_t start_infer_time = clock();

        qDebug() << "Doing Clas-Infer." << "\n";
        cls_ModelPredict((const uchar*)image.data, image.cols, image.rows, 3, pre_score, pre_category, pre_category_id);
        double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
        emit SetCostTime(cost_time);
    } catch (QException &e) {

        doing_Infer = false;
        qDebug() << "Finished Clas-Infer, but it is raise a exception." << "\n";

        emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

        return;
    }


    cvtColor(image, image, COLOR_BGR2RGB);

    float ratio = min(image.cols, image.rows) / 512.;
    int new_h = image.cols / ratio;
    int new_w = image.rows / ratio;
    cv::resize(image, image, cv::Size(new_h/4*4,new_w/4*4));

    if (image1 == nullptr)
    {
        image1 = new Mat(image.clone());
    }
    else
    {
        delete image1;
        image1 = new Mat(image.clone());
    }
    if (label1_image == nullptr)
    {
        label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                  image1->step, QImage::Format_RGB888);
    }
    else
    {
        delete label1_image;
        label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                  image1->step, QImage::Format_RGB888);
    }

    int color_[3] = { (int)(color_map[(pre_category_id[0] % 256) * 3]),
                      (int)(color_map[(pre_category_id[0] % 256) * 3 + 1]),
                      (int)(color_map[(pre_category_id[0] % 256) * 3 + 2]) };

    QString disscribe_str = makeLabelInfo(QString(pre_category), pre_category_id[0], pre_score[0]);
    int baseline[1];
    auto text_size = getTextSize(disscribe_str.toStdString(), FONT_HERSHEY_SIMPLEX,
                      1.0, 2, baseline);
    int text_left_downx = 0;
    int text_left_downy = 0 + text_size.height;

    putText(image, disscribe_str.toStdString(), Point(text_left_downx, text_left_downy),
            FONT_HERSHEY_SIMPLEX, 1.0,
            Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);


    if (image2 == nullptr)
    {
        image2 = new Mat(image.clone());
    }
    else
    {
        delete image2;
        image2 = new Mat(image.clone());
    }
    if (label2_image == nullptr)
    {
        label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                  image2->step, QImage::Format_RGB888);
    }
    else
    {
        delete label2_image;
        label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                  image2->step, QImage::Format_RGB888);
    }

    emit InferFinished(label1_image, label2_image);

    doing_Infer = false;
    qDebug() << "Finished Clas-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

}

void InferThread::Cls_Images()
{

    doing_Infer = true;

    for (int j = 0; j < images_path.count(); j++)
    {
        if (break_Infer)
        {
            doing_Infer = false;
            break_Infer = false;

            qDebug() << "Clas-Infer has Break." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

            return;
        }

        QString img_file = images_path[j];
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
            cls_ModelPredict((const uchar*)image.data, image.cols, image.rows, 3, pre_score, pre_category, pre_category_id);
            double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
            emit SetCostTime(cost_time);
        } catch (QException &e) {

            doing_Infer = false;
            qDebug() << "Finished Clas-Infer, but it is raise a exception." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

            return;
        }


        cvtColor(image, image, COLOR_BGR2RGB);

        float ratio = min(image.cols, image.rows) / 512.;
        int new_h = image.cols / ratio;
        int new_w = image.rows / ratio;
        cv::resize(image, image, cv::Size(new_h/4*4,new_w/4*4));

        if (image1 == nullptr)
        {
            image1 = new Mat(image.clone());
        }
        else
        {
            delete image1;
            image1 = new Mat(image.clone());
        }
        if (label1_image == nullptr)
        {
            label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                      image1->step, QImage::Format_RGB888);
        }
        else
        {
            delete label1_image;
            label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                      image1->step, QImage::Format_RGB888);
        }

        int color_[3] = { (int)(color_map[(pre_category_id[0] % 256) * 3]),
                          (int)(color_map[(pre_category_id[0] % 256) * 3 + 1]),
                          (int)(color_map[(pre_category_id[0] % 256) * 3 + 2]) };

        QString disscribe_str = makeLabelInfo(QString(pre_category), pre_category_id[0], pre_score[0]);
        int baseline[1];
        auto text_size = getTextSize(disscribe_str.toStdString(), FONT_HERSHEY_SIMPLEX,
                          1.0, 2, baseline);
        int text_left_downx = 0;
        int text_left_downy = 0 + text_size.height;

        putText(image, disscribe_str.toStdString(), Point(text_left_downx, text_left_downy),
                FONT_HERSHEY_SIMPLEX, 1.0,
                Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);

        if (image2 == nullptr)
        {
            image2 = new Mat(image.clone());
        }
        else
        {
            delete image2;
            image2 = new Mat(image.clone());
        }
        if (label2_image == nullptr)
        {
            label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                      image2->step, QImage::Format_RGB888);
        }
        else
        {
            delete label2_image;
            label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                      image2->step, QImage::Format_RGB888);
        }

        emit InferFinished(label1_image, label2_image);

        this->msleep(infer_Delay);
    }


    doing_Infer = false;
    qDebug() << "Finished Clas-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

}

void InferThread::Cls_Video()
{

    doing_Infer = true;

    VideoCapture cap = VideoCapture(video_path.toLocal8Bit().toStdString());
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

        if (break_Infer)
        {
            doing_Infer = false;
            break_Infer = false;

            qDebug() << "Clas-Infer has Break." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

            return;
        }

        float pre_score[1];
        int pre_category_id[1];
        char pre_category[200];

        try {
            clock_t start_infer_time = clock();

            qDebug() << "Doing Clas-Infer." << "\n";
            cls_ModelPredict((const uchar*)frame.data, frame.cols, frame.rows, 3, pre_score, pre_category, pre_category_id);
            double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
            emit SetCostTime(cost_time);
        } catch (QException &e) {

            doing_Infer = false;
            qDebug() << "Finished Clas-Infer, but it is raise a exception." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

            return;
        }


        cvtColor(frame, frame, COLOR_BGR2RGB);
        if (image1 == nullptr)
        {
            image1 = new Mat(frame.clone());
        }
        else
        {
            delete image1;
            image1 = new Mat(frame.clone());
        }
        if (label1_image == nullptr)
        {
            label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                      image1->step, QImage::Format_RGB888);
        }
        else
        {
            delete label1_image;
            label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                      image1->step, QImage::Format_RGB888);
        }

        int color_[3] = { (int)(color_map[(pre_category_id[0] % 256) * 3]),
                          (int)(color_map[(pre_category_id[0] % 256) * 3 + 1]),
                          (int)(color_map[(pre_category_id[0] % 256) * 3 + 2]) };

        QString disscribe_str = makeLabelInfo(QString(pre_category), pre_category_id[0], pre_score[0]);
        int baseline[1];
        auto text_size = getTextSize(disscribe_str.toStdString(), FONT_HERSHEY_SIMPLEX,
                          1.0, 2, baseline);
        int text_left_downx = 0;
        int text_left_downy = 0 + text_size.height;

        putText(frame, disscribe_str.toStdString(), Point(text_left_downx, text_left_downy),
                FONT_HERSHEY_SIMPLEX, 1.0,
                Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);

        if (image2 == nullptr)
        {
            image2 = new Mat(frame.clone());
        }
        else
        {
            delete image2;
            image2 = new Mat(frame.clone());
        }
        if (label2_image == nullptr)
        {
            label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                      image2->step, QImage::Format_RGB888);
        }
        else
        {
            delete label2_image;
            label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                      image2->step, QImage::Format_RGB888);
        }

        emit InferFinished(label1_image, label2_image);

        cap >> frame;
    }


    doing_Infer = false;
    qDebug() << "Finished Clas-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

}

void InferThread::Mask_Image()
{

    Mat image = imread(image_path.toLocal8Bit().toStdString());  //BGR

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

    doing_Infer = true;
    try {
        clock_t start_infer_time = clock();

        qDebug() << "Doing Mask-Infer." << "\n";
        mask_ModelPredict((const uchar*)image.data, image.cols, image.rows, 3, bboxs, out_image, bbox_num, labellist);
        double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
        emit SetCostTime(cost_time);
    } catch (QException &e) {

        doing_Infer = false;
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
                out3c_image.at<Vec3b>(i, j) = Vec3b(color_map[color_id * 3], color_map[color_id * 3 + 1], color_map[color_id * 3 + 2]);
        }
    }


    cvtColor(image, image, COLOR_BGR2RGB);
    if (image1 == nullptr)
    {
        image1 = new Mat(image.clone());
    }
    else
    {
        delete image1;
        image1 = new Mat(image.clone());
    }
    if (label1_image == nullptr)
    {
        label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                  image1->step, QImage::Format_RGB888);
    }
    else
    {
        delete label1_image;
        label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                  image1->step, QImage::Format_RGB888);
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

        if (score >= det_Threshold)
        {
            int color_[3] = { (int)(color_map[(categry_id % 256) * 3]),
                              (int)(color_map[(categry_id % 256) * 3 + 1]),
                              (int)(color_map[(categry_id % 256) * 3 + 2]) };

            QString disscribe_str = makeLabelInfo(label_list[i], categry_id, score);
            int baseline[1];
            auto text_size = getTextSize(disscribe_str.toStdString(), FONT_HERSHEY_SIMPLEX,
                              1.0, 2, baseline);
            int text_left_downx = left_topx;
            int text_left_downy = left_topy + text_size.height;

            rectangle(image, Point(left_topx, left_topy),
                      Point(right_downx, right_downy),
                      Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
            putText(image, disscribe_str.toStdString(), Point(text_left_downx, text_left_downy),
                    FONT_HERSHEY_SIMPLEX, 1.0,
                    Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
        }
    }

    if (image2 == nullptr)
    {
        image2 = new Mat(image.clone());
    }
    else
    {
        delete image2;
        image2 = new Mat(image.clone());
    }
    if (label2_image == nullptr)
    {
        label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                  image2->step, QImage::Format_RGB888);
    }
    else
    {
        delete label2_image;
        label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                  image2->step, QImage::Format_RGB888);
    }

    emit InferFinished(label1_image, label2_image);


    doing_Infer = false;
    qDebug() << "Finished Mask-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

}

void InferThread::Mask_Images()
{

    doing_Infer = true;

    for (int j = 0; j < images_path.count(); j++)
    {
        if (break_Infer)
        {
            doing_Infer = false;
            break_Infer = false;

            qDebug() << "Mask-Infer has Break." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

            return;
        }

        QString img_file = images_path[j];
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
            mask_ModelPredict((const uchar*)image.data, image.cols, image.rows, 3, bboxs, out_image, bbox_num, labellist);
            double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
            emit SetCostTime(cost_time);
        } catch (QException &e) {

            doing_Infer = false;
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
                    out3c_image.at<Vec3b>(i, j) = Vec3b(color_map[color_id * 3], color_map[color_id * 3 + 1], color_map[color_id * 3 + 2]);
            }
        }


        cvtColor(image, image, COLOR_BGR2RGB);
        if (image1 == nullptr)
        {
            image1 = new Mat(image.clone());
        }
        else
        {
            delete image1;
            image1 = new Mat(image.clone());
        }
        if (label1_image == nullptr)
        {
            label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                      image1->step, QImage::Format_RGB888);
        }
        else
        {
            delete label1_image;
            label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                      image1->step, QImage::Format_RGB888);
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

            if (score >= det_Threshold)
            {
                int color_[3] = { (int)(color_map[(categry_id % 256) * 3]),
                                  (int)(color_map[(categry_id % 256) * 3 + 1]),
                                  (int)(color_map[(categry_id % 256) * 3 + 2]) };

                QString disscribe_str = makeLabelInfo(label_list[i], categry_id, score);
                int baseline[1];
                auto text_size = getTextSize(disscribe_str.toStdString(), FONT_HERSHEY_SIMPLEX,
                                  1.0, 2, baseline);
                int text_left_downx = left_topx;
                int text_left_downy = left_topy + text_size.height;

                rectangle(image, Point(left_topx, left_topy),
                          Point(right_downx, right_downy),
                          Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
                putText(image, disscribe_str.toStdString(), Point(text_left_downx, text_left_downy),
                        FONT_HERSHEY_SIMPLEX, 1.0,
                        Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
            }
        }

        if (image2 == nullptr)
        {
            image2 = new Mat(image.clone());
        }
        else
        {
            delete image2;
            image2 = new Mat(image.clone());
        }
        if (label2_image == nullptr)
        {
            label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                      image2->step, QImage::Format_RGB888);
        }
        else
        {
            delete label2_image;
            label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                      image2->step, QImage::Format_RGB888);
        }

        emit InferFinished(label1_image, label2_image);

        this->msleep(infer_Delay);
    }

    doing_Infer = false;
    qDebug() << "Finished Mask-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

}

void InferThread::Mask_Video()
{
    doing_Infer = true;

    VideoCapture cap = VideoCapture(video_path.toLocal8Bit().toStdString());
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

        if (break_Infer)
        {
            doing_Infer = false;
            break_Infer = false;

            qDebug() << "Mask-Infer has Break." << "\n";

            emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

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
            mask_ModelPredict((const uchar*)frame.data, frame.cols, frame.rows, 3, bboxs, out_image, bbox_num, labellist);
            double cost_time = 1000 * (clock() - start_infer_time) / (double)CLOCKS_PER_SEC;
            emit SetCostTime(cost_time);
        } catch (QException &e) {

            doing_Infer = false;
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
                    out3c_image.at<Vec3b>(i, j) = Vec3b(color_map[color_id * 3], color_map[color_id * 3 + 1], color_map[color_id * 3 + 2]);
            }
        }

        cvtColor(frame, frame, COLOR_BGR2RGB);
        if (image1 == nullptr)
        {
            image1 = new Mat(frame.clone());
        }
        else
        {
            delete image1;
            image1 = new Mat(frame.clone());
        }
        if (label1_image == nullptr)
        {
            label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                      image1->step, QImage::Format_RGB888);
        }
        else
        {
            delete label1_image;
            label1_image = new QImage((const uchar*)image1->data, image1->cols, image1->rows,
                                      image1->step, QImage::Format_RGB888);
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

            if (score >= det_Threshold)
            {
                int color_[3] = { (int)(color_map[(categry_id % 256) * 3]),
                                  (int)(color_map[(categry_id % 256) * 3 + 1]),
                                  (int)(color_map[(categry_id % 256) * 3 + 2]) };

                QString disscribe_str = makeLabelInfo(label_list[i], categry_id, score);
                int baseline[1];
                auto text_size = getTextSize(disscribe_str.toStdString(), FONT_HERSHEY_SIMPLEX,
                                  1.0, 2, baseline);
                int text_left_downx = left_topx;
                int text_left_downy = left_topy + text_size.height;

                rectangle(frame, Point(left_topx, left_topy),
                          Point(right_downx, right_downy),
                          Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
                putText(frame, disscribe_str.toStdString(), Point(text_left_downx, text_left_downy),
                        FONT_HERSHEY_SIMPLEX, 1.0,
                        Scalar(color_[0], color_[1], color_[2]), 2, LINE_8);
            }
        }

        if (image2 == nullptr)
        {
            image2 = new Mat(frame.clone());
        }
        else
        {
            delete image2;
            image2 = new Mat(frame.clone());
        }
        if (label2_image == nullptr)
        {
            label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                      image2->step, QImage::Format_RGB888);
        }
        else
        {
            delete label2_image;
            label2_image = new QImage((const uchar*)image2->data, image2->cols, image2->rows,
                                      image2->step, QImage::Format_RGB888);
        }

        emit InferFinished(label1_image, label2_image);

        cap >> frame;
    }

    doing_Infer = false;
    qDebug() << "Finished Mask-Infer." << "\n";

    emit SetState_Btn_StopAndInfer(false, true);  // first is stop, second is infer

}
