---
comments: true
---

# PaddleX Pipelines (NPU)

## 1. Basic Pipelines

<table>
  <tr>
    <th width="10%">Pipeline Name</th>
    <th width="10%">Pipeline Modules</th>
    <th width="10%">Baidu AI Studio Community Experience URL</th>
    <th width="50%">Pipeline Introduction</th>
    <th width="20%">Applicable Scenarios</th>
  </tr>
  <tr>
    <td>Image Classification</td>
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
    <td>Object Detection</td>
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
    <td>Semantic Segmentation</td>
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
    <td>Instance Segmentation</td>
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
    <td rowspan="5">General OCR</td>
    <td>Text Detection</td>
    <td rowspan="5"><a href="https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent">Online Experience</a></td>
    <td rowspan="5">OCR (Optical Character Recognition) is a technology that converts text in images into editable text. It is widely used in document digitization, information extraction, and data processing. OCR can recognize printed text, handwritten text, and even certain types of fonts and symbols. General OCR is used to solve text recognition tasks, extracting text information from images and outputting it in text form. PP-OCRv4 is an end-to-end OCR system that can achieve millisecond-level accurate text prediction on CPUs, reaching open-source SOTA in general scenarios. Based on this project, many developers from academia, industry, and research have quickly implemented multiple OCR applications, covering various fields such as general, manufacturing, finance, and transportation.</td>
    <td rowspan="5">
    <ul>
        <li>License plate recognition in intelligent security</li>
        <li>Recognition of house numbers and other information</li>
        <li>Digitization of paper documents</li>
        <li>Recognition of ancient characters in cultural heritage</li>
    </ul>
    </td>
</tr>
<tr>
    <td>Text Recognition</td>
</tr>
<tr>
    <td>Document Image Orientation Classification </td>
</tr>
<tr>
    <td>Text Image Unwarping </td>
</tr>
<tr>
    <td>Text Line Orientation Classification </td>
</tr>
<tr>
    <td rowspan="6">General Table Recognition</td>
    <td>Table Structure Recognition</td>
    <td rowspan="6"><a href="https://aistudio.baidu.com/community/app/91661/webUI">Online Experience</a></td>
    <td rowspan="6">Table recognition is a technology that automatically identifies and extracts table content and structure from documents or images. It is widely used in data entry, information retrieval, and document analysis. By using computer vision and machine learning algorithms, table recognition can convert complex table information into an editable format, facilitating further processing and analysis by users.</td>
    <td rowspan="6">
    <ul>
        <li>Processing of bank statements</li>
        <li>Recognition and extraction of indicators in medical reports</li>
        <li>Extraction of table information in contracts</li>
    </ul>
    </td>
</tr>
<tr>
    <td>Text Detection</td>
</tr>
<tr>
    <td>Text Recognition</td>
</tr>
<tr>
    <td>Layout Detection </td>
</tr>
<tr>
    <td>Doc Img Orientation Classification </td>
</tr>
<tr>
    <td>Text Image Unrapping </td>
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
<tr>
    <td>Multi-label Image Classification</td>
    <td>Multi-label Image Classification</td>
    <td><a href="https://aistudio.baidu.com/community/app/387974/webUI?source=appCenter">Online Experience</a></td>
    <td>Image multi-label classification is a technology that assigns an image to multiple related categories simultaneously. It is widely used in image tagging, content recommendation, and social media analysis. It can identify multiple objects or features present in an image, such as both "dog" and "outdoor" labels in a single picture. By using deep learning models, image multi-label classification can automatically extract image features and perform accurate classification to provide more comprehensive information for users. This technology is significant in applications like intelligent search engines and automatic content generation.</td>
    <td>
    <ul>
        <li>Medical image diagnosis</li>
        <li>Complex scene recognition</li>
        <li>Multi-target monitoring</li>
        <li>Product attribute recognition</li>
        <li>Ecological environment monitoring</li>
        <li>Security monitoring</li>
        <li>Disaster warning</li>
      </ul>
      </td>
  </tr>
  <tr>
    <td>Small Object Detection</td>
    <td>Small Object Detection</td>
    <td><a href="https://aistudio.baidu.com/community/app/387975/webUI?source=appCenter">Online Experience</a></td>
    <td>Small object detection is a technology specifically for identifying small objects in images. It is widely used in surveillance, autonomous driving, and satellite image analysis. It can accurately find and classify small-sized objects like pedestrians, traffic signs, or small animals in complex scenes. By using deep learning algorithms and optimized convolutional neural networks, small object detection can effectively enhance the recognition ability of small objects, ensuring that important information is not missed in practical applications. This technology plays an important role in improving safety and automation levels.</td>
    <td>
  <ul>
    <li>Pedestrian detection in autonomous vehicles</li>
    <li>Identification of small buildings in satellite images</li>
    <li>Detection of small traffic signs in intelligent transportation systems</li>
    <li>Identification of small intruding objects in security surveillance</li>
    <li>Detection of small defects in industrial inspection</li>
    <li>Monitoring of small animals in drone images</li>
  </ul>
</td>
  </tr>
  <tr>
    <td>Image Anomaly Detection</td>
    <td>Image Anomaly Detection</td>
    <td>None</td>
    <td>Image anomaly detection is a technology that identifies images that deviate from or do not conform to normal patterns by analyzing their content. It is widely used in industrial quality inspection, medical image analysis, and security surveillance. By using machine learning and deep learning algorithms, image anomaly detection can automatically identify potential defects, anomalies, or abnormal behavior in images, helping us detect problems and take appropriate measures promptly. Image anomaly detection systems are designed to automatically detect and label abnormal situations in images to improve work efficiency and accuracy.</td>
    <td>
    <ul>
    <li>Industrial quality control</li>
    <li>Medical image analysis</li>
    <li>Anomaly detection in surveillance videos</li>
    <li>Identification of violations in traffic monitoring</li>
    <li>Obstacle detection in autonomous driving</li>
    <li>Agricultural pest and disease monitoring</li>
    <li>Pollutant identification in environmental monitoring</li>
  </ul></td>
  </tr>
  <tr>
    <td rowspan="10">General Layout Parsing</td>
    <td>Layout Detection</td>
    <td rowspan="10">None</td>
    <td rowspan="10">Layout parsing is a technology that extracts structured information from document images, primarily used to convert complex document layouts into machine-readable data formats. This technology is widely applied in document management, information extraction, and data digitization. By combining Optical Character Recognition (OCR), image processing, and machine learning algorithms, layout parsing can identify and extract text blocks, headings, paragraphs, images, tables, and other layout elements from documents. The process typically includes three main steps: layout analysis, element analysis, and data formatting, ultimately generating structured document data to enhance the efficiency and accuracy of data processing.</td>
    <td rowspan="10">
        <ul>
            <li>Analysis of financial and legal documents</li>
            <li>Digitization of historical documents and archives</li>
            <li>Automated form filling</li>
            <li>Page structure parsing</li>
        </ul>
    </td>
</tr>
<tr>
    <td>Layout Detection Module</td>
</tr>
<tr>
    <td>Text Detection Module</td>
</tr>
<tr>
    <td>Text Recognition Module</td>
</tr>
<tr>
    <td>Doc Img Orientation Classification</td>
</tr>
<tr>
    <td>Text Image Unrapping</td>
</tr>
<tr>
    <td>Table Structure Recognition</td>
</tr>
<tr>
    <td>Text Line Orientation Classification</td>
</tr>
<tr>
    <td>Formula Recognition</td>
</tr>
<tr>
    <td>Seal Text Detection</td>
</tr>
<tr>
    <td rowspan="4">Formula Recognition</td>
    <td>Formula Recognition</td>
    <td rowspan="4"><a href="https://aistudio.baidu.com/community/app/387976/webUI?source=appCenter">Online Experience</a></td>
    <td rowspan="4">Formula recognition is a technology that automatically identifies and extracts LaTeX formula content and structure from documents or images. It is widely used in document editing and data analysis in fields such as mathematics, physics, and computer science. By using computer vision and machine learning algorithms, formula recognition can convert complex mathematical formula information into editable LaTeX format, facilitating further processing and analysis by users.</td>
    <td rowspan="4">
        <ul>
            <li>Document digitization and retrieval</li>
            <li>Formula search engine</li>
            <li>Formula editor</li>
            <li>Automated typesetting</li>
        </ul>
    </td>
</tr>
<tr>
    <td>Layout Detection Module </td>
</tr>
<tr>
    <td>Doc Img Orientation Classification </td>
</tr>
<tr>
    <td>Text Image Unrapping</td>
</tr>
<tr>
    <td rowspan="5">Seal Text Recognition</td>
    <td>Seal Text Detection</td>
    <td rowspan="5"><a href="https://aistudio.baidu.com/community/app/387977/webUI?source=appCenter">Online Experience</a></td>
    <td rowspan="5">Seal text recognition is a technology that automatically extracts and identifies seal content from documents or images. Seal text recognition is a part of document processing and is useful in many scenarios, such as contract comparison, inventory audit, and invoice reimbursement review.</td>
    <td rowspan="5">
        <ul>
            <li>Contract and agreement verification</li>
            <li>Check processing</li>
            <li>Loan approval</li>
            <li>Legal document management</li>
        </ul>
    </td>
</tr>
<tr>
    <td>Text Recognition</td>
</tr>
<tr>
    <td>Layout Detection </td>
</tr>
<tr>
    <td>Doc Img Orientation Classification </td>
</tr>
<tr>
    <td>Text Image Unrapping</td>
</tr>
<tr>
    <td rowspan = 2>General Image Recognition</td>
    <td>Mainbody Detection</td>
    <td rowspan = 2>None</td>
    <td rowspan = 2>The general image recognition pipeline is designed to address open-domain target localization and recognition issues. It can effectively identify and differentiate various target objects in different environments and conditions, making it widely applicable in autonomous driving, intelligent security, medical image analysis, and industrial automation, among other fields.</td>
    <td rowspan = 2>
    <ul>
        <li>Automated Identity Verification</li>
        <li>Unmanned Retail</li>
        <li>Autonomous Driving</li>
      </ul>
      </td>
  </tr>
  <tr>
    <td>Image Features</td>
  </tr>
  <tr>
    <td rowspan = 2>Pedestrian Attribute Recognition</td>
    <td>Pedestrian Detection</td>
    <td rowspan = 2>None</td>
    <td rowspan = 2>Pedestrian attribute recognition is a key function in computer vision systems used to locate and tag specific features of pedestrians in images or videos, such as gender, age, clothing color, and style.</td>
    <td rowspan = 2>
    <ul>
        <li>Smart City</li>
        <li>Security Monitoring</li>
      </ul>
      </td>
  </tr>
  <tr>
    <td>Pedestrian Attribute Recognition</td>
  </tr>
  <tr>
    <td rowspan = 2>Vehicle Attribute Recognition</td>
    <td>Vehicle Detection</td>
    <td rowspan = 2>None</td>
    <td rowspan = 2>Vehicle attribute recognition is an important component of computer vision systems. Its main task is to locate and tag specific attributes of vehicles in images or videos, such as vehicle type, color, and license plate number. This task not only requires accurate detection of vehicles but also the recognition of detailed attribute information for each vehicle.</td>
    <td rowspan = 2>
    <ul>
        <li>Intelligent Parking</li>
        <li>Traffic Management</li>
        <li>Autonomous Driving</li>
      </ul>
      </td>
  </tr>
  <tr>
    <td>Vehicle Attribute Recognition</td>
  </tr>
<tr>
    <td rowspan="2">Document Image Preprocessing</td>
    <td>Doc Img Orientation Classification</td>
    <td rowspan="2">Not Available</td>
    <td rowspan="2">Document image preprocessing is a key step in document analysis and recognition, aiming to optimize document images through a series of technical means to improve the accuracy and efficiency of subsequent processing. Document image preprocessing includes operations such as orientation classification, text rectification, noise removal, and binarization, which can effectively improve image quality, correct document orientation, and remove interference factors. This technology is widely used in document scanning, OCR text recognition, and electronic document generation.</td>
    <td rowspan="2">
    <ul>
        <li>Automatic orientation correction in document scanners</li>
        <li>Text image optimization in OCR systems</li>
        <li>Image restoration in historical document digitization</li>
    </ul>
    </td>
</tr>
<tr>
    <td>Text Image Unrapping</td>
</tr>
</table>

## 2. Featured Pipelines
Not supported yet, please stay tuned!
