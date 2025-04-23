---
comments: true
---

# PaddleX Pipelines (NPU)

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
        <li>Real-time monitoring of defective products on pipelines</li>
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
    <td rowspan = 7>PP-ChatOCRv3</td>
    <td>Table Structure Recognition</td>
    <td rowspan = 7><a href="https://aistudio.baidu.com/community/app/182491/webUI?source=appCenter">Online Experience</a></td>
    <td rowspan = 7>Document Image Scene Information Extraction v3 (PP-ChatOCRv3-doc) is a PaddlePaddle-specific intelligent document and image analysis solution that integrates LLM and OCR technologies to solve common complex document information extraction challenges such as layout analysis, rare characters, multi-page PDFs, tables, and seal recognition. By integrating the Wenxin large model, it combines vast data and knowledge, providing high accuracy and wide applicability. The open-source version supports local experience and deployment, and fine-tuning training for each module.</td>
    <td rowspan="7">
  <ul>
    <li>Construction of knowledge graphs</li>
    <li>Detection of information related to specific events in online news and social media</li>
    <li>Extraction and analysis of key information in academic literature (especially in scenarios requiring recognition of seals, distorted images, and more complex tables)</li>
  </ul>
</td>
  </tr>
  <tr>
    <td>Layout Detection</td>
  </tr>
  <tr>
    <td>Text Detection</td>
  </tr>
  <tr>
    <td>Text Recognition</td>
  </tr>
  <tr>
    <td>Seal Text Detection</td>
  </tr>
  <tr>
    <td>Text Image Unrapping</td>
  </tr>
  <tr>
    <td>Document Image Orientation Classification</td>
  </tr>
  <tr>
      <td rowspan="8">PP-ChatOCRv4</td>
      <td>Table Structure Recognition</td>
      <td rowspan="8">Coming Soon</td>
      <td rowspan="8">Document Scene Information Extraction v4 (PP-ChatOCRv4) is a PaddlePaddle-featured intelligent analysis solution for documents and images, combining LLM, MLLM, and OCR technologies. Based on PP-ChatOCRv3, it optimizes common complex document information extraction challenges such as layout analysis, rare characters, multi-page PDFs, tables, and seal recognition. It integrates massive data and knowledge with the Ernie model, achieving high accuracy and wide applicability. This pipeline also provides flexible service deployment methods, supporting deployment on various hardware. Furthermore, it offers secondary development capabilities, allowing you to train and optimize on your own dataset, and the trained model can be seamlessly integrated.</td>
      <td rowspan="8">
          <ul>
              <li>Knowledge Graph Construction</li>
              <li>Detection of Information Related to Specific Events in Online News and Social Media</li>
              <li>Extraction and Analysis of Key Information in Academic Literature (especially scenarios requiring recognition of seals, distorted images, and more complex tables)</li>
          </ul>
      </td>
  </tr>
  <tr>
      <td>Layout Detection</td>
  </tr>
  <tr>
      <td>Text Detection</td>
  </tr>
  <tr>
      <td>Text Recognition</td>
  </tr>
  <tr>
      <td>Seal Text Detection</td>
  </tr>
  <tr>
      <td>Text Image Unrapping</td>
  </tr>
  <tr>
      <td>Document Image Orientation Classification</td>
  </tr>
  <tr>
      <td>Document-based Vision-Language Model</td>
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
Not supported yet, please stay tuned!
