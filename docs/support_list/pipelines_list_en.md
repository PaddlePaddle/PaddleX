[简体中文](pipelines_list.md) | English

# PaddleX Pipelines (CPU/GPU)

## 1. Basic Pipelines

<table>
  <tr>
    <th width="10%">Pipeline Name</th>
    <th width="10%">Pipeline Modules</th>
    <th width="10%">Baidu AIStudio Community Experience URL</th>
    <th width="50%">Pipeline Introduction</th>
    <th width="20%">Applicable Scenarios</th>
  </tr>
  <tr>
    <td>General Image Classification</td>
    <td>Image Classification</td>
    <td><a href="https://aistudio.baidu.com/community/app/100061/webUI">Online Experience</a></td>
    <td>Image classification is a technique that assigns images to predefined categories. It is widely used in object recognition, scene understanding, and automatic annotation. Image classification can identify various objects such as animals, plants, traffic signs, etc., and categorize them based on their features. By leveraging deep learning models, image classification can automatically extract image features and perform accurate classification. The General Image Classification Pipeline is designed to solve image classification tasks for given images.</td>
    <td>
      <ul>
        <li>Automatic classification and recognition of product images</li>
        <li>Real-time monitoring of defective products on production lines</li>
        <li>Personnel recognition in security surveillance</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>General Object Detection</td>
    <td>Object Detection</td>
    <td><a href="https://aistudio.baidu.com/community/app/70230/webUI">Online Experience</a></td>
    <td>Object detection aims to identify the categories and locations of multiple objects in images or videos by generating bounding boxes to mark these objects. Unlike simple image classification, object detection not only recognizes what objects are in the image, such as people, cars, and animals, but also accurately determines the specific location of each object, usually represented by a rectangular box. This technology is widely used in autonomous driving, surveillance systems, and smart photo albums, relying on deep learning models (e.g., YOLO, Faster R-CNN) that efficiently extract features and perform real-time detection, significantly enhancing the computer's ability to understand image content.</td>
    <td>
      <ul>
        <li>Tracking moving objects in video surveillance</li>
        <li>Vehicle detection in autonomous driving</li>
        <li>Defect detection in industrial manufacturing</li>
        <li>Shelf product detection in retail</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>General Semantic Segmentation</td>
    <td>Semantic Segmentation</td>
    <td><a href="https://aistudio.baidu.com/community/app/100062/webUI?source=appCenter">Online Experience</a></td>
    <td>Semantic segmentation is a computer vision technique that assigns each pixel in an image to a specific category, enabling detailed understanding of image content. Semantic segmentation not only identifies the types of objects in an image but also classifies each pixel, allowing entire regions of the same category to be marked. For example, in a street scene image, semantic segmentation can distinguish pedestrians, cars, sky, and roads at the pixel level, forming a detailed label map. This technology is widely used in autonomous driving, medical image analysis, and human-computer interaction, often relying on deep learning models (e.g., FCN, U-Net) that use Convolutional Neural Networks (CNNs) to extract features and achieve high-precision pixel-level classification, providing a foundation for further intelligent analysis.</td>
    <td>
      <ul>
        <li>Analysis of satellite images in Geographic Information Systems</li>
        <li>Segmentation of obstacles and passable areas in robot vision</li>
        <li>Separation of foreground and background in film production</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>General Instance Segmentation</td>
    <td>Instance Segmentation</td>
    <td><a href="https://aistudio.baidu.com/community/app/100063/webUI">Online Experience</a></td>
    <td>Instance segmentation is a computer vision task that identifies object categories in images and distinguishes the pixels of different instances within the same category, enabling precise segmentation of each object. Instance segmentation can separately mark each car, person, or animal in an image, ensuring they are processed independently at the pixel level. For example, in a street scene image with multiple cars and pedestrians, instance segmentation can clearly separate the contours of each car and person, forming multiple independent region labels. This technology is widely used in autonomous driving, video surveillance, and robot vision, often relying on deep learning models (e.g., Mask R-CNN) that use CNNs for efficient pixel classification and instance differentiation, providing powerful support for understanding complex scenes.</td>
    <td>
      <ul>
        <li>Crowd counting in malls</li>
        <li>Counting crops or fruits in agricultural intelligence</li>
        <li>Selecting and segmenting specific objects in image editing</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td rowspan = 2>General OCR</td>
    <td >Text Detection</td>
    <td rowspan = 2><a href="https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent">Online Experience</a></td>
    <td rowspan = 2>OCR (Optical Character Recognition) is a technology that converts text in images into editable text. It is widely used in document digitization, information extraction, and data processing. OCR can recognize printed text, handwritten text, and even certain types of fonts and symbols. The General OCR Pipeline is designed to solve text recognition tasks, extracting text information from images and outputting it in text form. PP-OCRv4 is an end-to-end OCR system that achieves millisecond-level text content prediction on CPUs, achieving state-of-the-art (SOTA) performance in general scenarios. Based on this project, developers from academia, industry, and research have quickly implemented various OCR applications covering general, manufacturing, finance, transportation.</td>
    <td rowspan = 2>
      <ul>
        <li>Document digitization</li>
        <li>Information extraction</li>
        <li>Data processing</li>
      </ul>
    </td>
  </tr>
    <tr>
    <td>Text Recognition</td>
  </tr>
<tr>
        <td rowspan = 4>General Table Recognition</td>
        <td>Layout Detection</td>
        <td rowspan = 4><a href="https://aistudio.baidu.com/community/app/91661/webUI">Online Experience</a></td>
        <td rowspan = 4>Table recognition is a technology that automatically identifies and extracts table content and its structure from documents or images. It is widely used in data entry, information retrieval, and document analysis. By leveraging computer vision and machine learning algorithms, table recognition can convert complex table information into editable formats, facilitating further data processing and analysis by users</td>
<td rowspan = 4>
    <ul>
        <li>Processing of bank statements</li>
        <li>recognition and extraction of various indicators in medical reports</li>
        <li>extraction of tabular information from contracts</li>
      </ul>
      </td>
   </tr>
  <tr>
    <td>Table Structure Recognition </td>
  </tr>
  <tr>
    <td>Text Detection</td>
  </tr>
  <tr>
    <td>Text Recognition</td>
  </tr>
    <tr>
        <td>Time Series Forecasting</td>
        <td>Time Series Forecasting Module</td>
        <td><a href="https://aistudio.baidu.com/community/app/105706/webUI?source=appCenter">Online Experience</a></td>
        <td>Time series forecasting is a technique that utilizes historical data to predict future trends by analyzing patterns in time series data. It is widely applied in financial markets, weather forecasting, and sales prediction. Time series forecasting typically employs statistical methods or deep learning models (such as LSTM, ARIMA, etc.), which can handle time dependencies in data to provide accurate predictions, assisting decision-makers in better planning and response. This technology plays a crucial role in many industries, including energy management, supply chain optimization, and market analysis</td>
        <td>
    <ul>
        <li>Stock prediction</li>
        <li>climate forecasting</li>
        <li>disease spread prediction</li>
        <li>energy demand forecasting</li>
        <li>traffic flow prediction</li>
        <li>product lifecycle prediction</li>
        <li>electric load forecasting</li>
      </ul>
      </td>
    </tr>
    <tr>
        <td>Time Series Anomaly Detection</td>
        <td>Time Series Anomaly Detection Module</td>
        <td><a href="https://aistudio.baidu.com/community/app/105706/webUI?source=appCenter">Online Experience</a></td>
        <td>Time series anomaly detection is a technique that identifies abnormal patterns or behaviors in time series data. It is widely used in network security, device monitoring, and financial fraud detection. By analyzing normal trends and patterns in historical data, it discovers events that significantly differ from expected behaviors, such as sudden increases in network traffic or unusual transaction activities. Time series anomaly detection often employs statistical methods or machine learning algorithms (like Isolation Forest, LSTM, etc.), which can automatically identify anomalies in data, providing real-time alerts to enterprises and organizations to help promptly address potential risks and issues. This technology plays a vital role in ensuring system stability and security</td>
        <td>
    <ul>
        <li>Financial fraud detection</li>
        <li>network intrusion detection</li>
        <li>equipment failure detection</li>
        <li>industrial production anomaly detection</li>
        <li>stock market anomaly detection</li>
        <li>power system anomaly detection</li>
      </ul>
      </td>
    </tr>
    <tr>
        <td>Time Series Classification</td>
        <td>Time Series Classification Module</td>
        <td><a href="https://aistudio.baidu.com/community/app/105707/webUI?source=appCenter">Online Experience</a></td>
        <td>Time series classification is a technique that categorizes time series data into predefined classes. It is widely applied in behavior recognition, speech recognition, and financial trend analysis. By analyzing features that vary over time, it identifies different patterns or events, such as classifying a speech signal as "greeting" or "request" or dividing stock price movements into "rising" or "falling." Time series classification typically utilizes machine learning and deep learning models, effectively capturing time dependencies and variation patterns to provide accurate classification labels for data. This technology plays a key role in intelligent monitoring, voice assistants, and market forecasting applications</td>
            <td>
    <ul>
        <li>Electrocardiogram Classification</li>
        <li>Stock Market Behavior Classification</li>
        <li>Electroencephalogram Classification</li>
        <li>Emotion Classification</li>
        <li>Traffic Condition Classification</li>
        <li>Network Traffic Classification</li>
        <li>Equipment Operating Condition Classification</li>
      </ul>
      </td>
</table>

## 2. Featured Pipelines

<table>
  <tr>
    <th width="10%">Pipeline Name</th>
    <th width="10%">Pipeline Modules</th>
    <th width="10%">Baidu AIStudio Community Experience Link</th>
    <th width="50%">Pipeline Introduction</th>
    <th width="20%">Applicable Scenarios</th>
  </tr>
  <tr>
    <td>Semi-supervised Learning for Large Models - Image Classification</td>
    <td>Semi-supervised Learning for Large Models - Image Classification</td>
    <td><a href="https://aistudio.baidu.com/community/app/100061/webUI">Online Experience</a></td>
    <td>Image classification is a technique that assigns images to predefined categories. It is widely used in object recognition, scene understanding, and automatic annotation. Image classification can identify various objects such as animals, plants, traffic signs, etc., and categorize them based on their features. By leveraging deep learning models, image classification can automatically extract image features and perform accurate classification. The general image classification pipeline is designed to solve image classification tasks for given images.</td>
    <td>
      <ul>
        <li>Commodity image classification</li>
        <li>Artwork style classification</li>
        <li>Crop disease and pest identification</li>
        <li>Animal species recognition</li>
        <li>Classification of land, water bodies, and buildings in satellite remote sensing images</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td >Semi-supervised Learning for Large Models - Object Detection</td>
    <td>Semi-supervised Learning for Large Models - Object Detection</td>
    <td><a href="https://aistudio.baidu.com/community/app/70230/webUI">Online Experience</a></td>
    <td>The semi-supervised learning for large models - object detection pipeline is a unique offering from PaddlePaddle. It utilizes a joint training approach with large and small models, leveraging a small amount of labeled data and a large amount of unlabeled data to enhance model accuracy, significantly reducing the costs of manual model iteration and data annotation. The figure below demonstrates the performance of this pipeline on the COCO dataset with 10% labeled data. After training with this pipeline, on COCO 10% labeled data + 90% unlabeled data, the large model (RT-DETR-H) achieves an 8.4% higher accuracy (47.7% -> 56.1%), setting a new state-of-the-art (SOTA) for this dataset. The small model (PicoDet-S) also achieves over 10% higher accuracy (18.3% -> 28.8%) compared to direct training.</td>
    <td>
      <ul>
        <li>Pedestrian, vehicle, and traffic sign detection in autonomous driving</li>
        <li>Enemy facility and equipment detection in military reconnaissance</li>
        <li>Seabed organism detection in deep-sea exploration</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td rowspan = 2>Semi-supervised Learning for Large Models - OCR</td>
    <td>Text Detection</td>
    <td rowspan = 2><a href="https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent">Online Experience</a></td>
    <td rowspan = 2>The semi-supervised learning for large models - OCR pipeline is a unique OCR training pipeline from PaddlePaddle. It consists of a text detection model and a text recognition model working in series. The input image is first processed by the text detection model to obtain and rectify all text line bounding boxes, which are then fed into the text recognition model to generate OCR text results. In the text recognition part, a joint training approach with large and small models is adopted, utilizing a small amount of labeled data and a large amount of unlabeled data to enhance model accuracy, significantly reducing the costs of manual model iteration and data annotation. The figure below shows the effects of this pipeline in two OCR application scenarios, demonstrating significant improvements for both large and small models in different contexts.</td>
    <td rowspan = 2>
      <ul>
        <li>Digitizing paper documents</li>
        <li>Reading and verifying personal information on IDs, passports, and driver's licenses</li>
        <li>Recognizing product information in retail</li>
      </ul>
    </td>
  </tr>
    <tr>
      <td>Large Model Semi-supervised Learning - Text Recognition</td>
    </tr>
  <tr>
    <td rowspan = 2>General Scene Information Extraction v2</td>
    <td>Text Detection</td>
    <td rowspan = 2><a href="https://aistudio.baidu.com/community/app/91662?source=appCenter">Online Experience</a></td>
    <td rowspan = 2>The General Scene Information Extraction Pipeline (PP-ChatOCRv2-common) is a unique intelligent analysis solution for complex documents from PaddlePaddle. It combines Large Language Models (LLMs) and OCR technology, leveraging the Wenxin Large Model to integrate massive data and knowledge, achieving high accuracy and wide applicability. The system flow of PP-ChatOCRv2-common is as follows: Input the prediction image, send it to the general OCR system, predict text through text detection and text recognition models, perform vector retrieval between the predicted text and user queries to obtain relevant text information, and finally pass these text information to the prompt generator to recombine them into prompts for the Wenxin Large Model to generate prediction results.</td>
    <td rowspan = 2>
      <ul>
        <li>Key information extraction from various scenarios such as ID cards, bank cards, household registration books, train tickets, and paper invoices</li>
      </ul>
    </td>
  </tr>
      <tr>
      <td>Text Recognition</td>
    </tr>
</table>
