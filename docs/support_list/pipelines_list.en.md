---
comments: true
---

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
    <td>Image Classification</td>
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
    <td rowspan = 7>Document Scene Information Extraction v3</td>
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
    <td>Layout Area Detection</td>
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
    <td>Text Image Correction</td>
  </tr>
  <tr>
    <td>Document Image Orientation Classification</td>
  </tr>
  <tr>
    <td rowspan = 2>OCR</td>
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
        <td rowspan = 4>Table Recognition</td>
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
<tr>
    <td>Multi-label Image Classification</td>
    <td>Multi-label Image Classification</td>
    <td>None</td>
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
    <td>None</td>
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
    <td rowspan = 8>Layout Parsing</td>
    <td>Table Structure Recognition</td>
    <td rowspan = 8>None</td>
    <td rowspan = 8>Layout analysis is a technology for extracting structured information from document images, primarily used to convert complex document layouts into machine-readable data formats. This technology has wide applications in document management, information extraction, and data digitization. By combining optical character recognition (OCR), image processing, and machine learning algorithms, layout analysis can identify and extract text blocks, titles, paragraphs, images, tables, and other layout elements from documents. This process typically includes three main steps: layout analysis, element analysis, and data formatting, ultimately generating structured document data that enhances data processing efficiency and accuracy.</td>
    <td rowspan="8">
  <ul>
    <li>Financial and legal document analysis</li>
    <li>Digitization of historical documents and archives</li>
    <li>Automated form filling</li>
    <li>Page structure analysis</li>
  </ul>
</td>
  </tr>
  <tr>
    <td>Layout Area Detection</td>
  </tr>
  <tr>
    <td>Text Detection</td>
  </tr>
  <tr>
    <td>Text Recognition</td>
  </tr>
  <tr>
    <td>Formula Recognition</td>
  </tr>
  <tr>
    <td>Seal Text Detection</td>
  </tr>
  <tr>
    <td>Text Image Correction</td>
  </tr>
  <tr>
    <td>Document Image Orientation Classification</td>
  </tr>
    <tr>
    <td rowspan = 2>Formula Recognition</td>
    <td>Layout Area Detection</td>
    <td rowspan = 2>None</td>
    <td rowspan = 2>Formula recognition is a technology that automatically identifies and extracts LaTeX formula content and its structure from documents or images. It is widely used in document editing and data analysis in fields such as mathematics, physics, and computer science. By using computer vision and machine learning algorithms, formula recognition can convert complex mathematical formula information into an editable LaTeX format, facilitating further data processing and analysis by users.</td>
    <td rowspan = 2>
    <ul>
        <li>Document digitization and retrieval</li>
        <li>Formula search engine</li>
        <li>Formula editor</li>
        <li>Automated typesetting</li>
      </ul>
      </td>
  </tr>
  <tr>
    <td>Formula Recognition</td>
  </tr>
  <tr>
    <td rowspan = 3>Seal Text Recognition</td>
    <td>Layout Area Detection</td>
    <td rowspan = 3>None</td>
    <td rowspan = 3>Seal text recognition is a technology that automatically extracts and recognizes seal content from documents or images. Recognizing seal text is part of document processing and has applications in many scenarios, such as contract comparison, inventory audit, and invoice reimbursement audit.</td>
    <td rowspan = 3>
    <ul>
        <li>Contract and agreement validation</li>
        <li>Check processing</li>
        <li>Loan approval</li>
        <li>Legal document management</li>
      </ul>
      </td>
  </tr>
  <tr>
    <td>Seal Text Detection</td>
  </tr>
  <tr>
    <td>Text Recognition</td>
  </tr>
<tr>
    <td rowspan = 2>General Image Recognition</td>
    <td>Subject Detection</td>
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
    <td rowspan = 2>Face Recognition</td>
    <td>Face Detection</td>
    <td rowspan = 2>None</td>
    <td rowspan = 2>The facial recognition task is an important component of the computer vision field, aiming to realize automatic personal identity recognition through the analysis and comparison of facial features.</td>
    <td rowspan = 2>
    <ul>
        <li>Security Authentication</li>
        <li>Monitoring Systems</li>
        <li>Social Media</li>
      </ul>
      </td>
  </tr>
  <tr>
    <td>Face Features</td>
  </tr>
  <tr>
    <td>3D Multimodal Fusion Detection</td>
    <td>3D Multimodal Fusion Detection</td>
    <td>Not Available</td>
    <td>3D multimodal fusion detection is a technology that combines multiple data modalities (such as LiDAR, cameras, and millimeter-wave radar) to detect targets in three-dimensional space. It leverages the strengths of different modalities to achieve more accurate target localization, classification, and tracking. Through deep learning algorithms, this technology can process complex 3D scenes, identify vehicles, pedestrians, obstacles, and other targets, and provide key support for fields such as autonomous driving, intelligent transportation, and robot navigation.</td>
    <td>
    <ul>
        <li>Obstacle detection and avoidance in autonomous vehicles</li>
        <li>Traffic flow monitoring in intelligent transportation systems</li>
        <li>Object recognition and grasping in industrial robots</li>
    </ul>
    </td>
</tr>
<tr>
    <td rowspan="2">Human Keypoint Detection</td>
    <td>Pedestrian Detection</td>
    <td rowspan="2">Not Available</td>
    <td rowspan="2">Human keypoint detection is an important task in computer vision, aiming to locate specific parts of the human body (such as the head, shoulders, elbows, knees, etc.) through image or video data. By analyzing the geometric structure and appearance features of the human body, this technology can capture human posture and movements in real-time and is widely used in human-computer interaction, motion analysis, and virtual reality.</td>
    <td rowspan="2">
    <ul>
        <li>Movement guidance in smart fitness applications</li>
        <li>Character movement capture in virtual reality</li>
        <li>Abnormal behavior analysis in security surveillance</li>
      </ul>
      </td>
</tr>
<tr>
    <td>Keypoint Detection</td>
</tr>
<tr>
    <td>Open-Vocabulary Detection</td>
    <td>Open-Vocabulary Detection</td>
    <td>Not Available</td>
    <td>Open-vocabulary detection is an emerging computer vision technology aimed at enabling models to recognize and understand new categories or vocabulary not seen during training. Unlike traditional object detection, open-vocabulary detection does not rely on large amounts of labeled data but instead combines pre-trained language models and visual features to quickly recognize and understand unknown categories. This technology has broad application prospects in dynamic environment object detection, image classification, and intelligent robots.</td>
    <td>
    <ul>
        <li>Recognition of unknown obstacles in autonomous driving</li>
        <li>Abnormal behavior detection in intelligent security</li>
        <li>Target exploration by intelligent robots in complex environments</li>
    </ul>
    </td>
</tr>
<tr>
    <td>Open-Vocabulary Segmentation</td>
    <td>Open-Vocabulary Segmentation</td>
    <td>Not Available</td>
    <td>Open-vocabulary segmentation is a cutting-edge computer vision technology aimed at performing pixel-level semantic segmentation of unknown categories in images. Unlike traditional segmentation methods limited to labeled categories, open-vocabulary segmentation combines pre-trained language models and visual features to dynamically recognize and segment new categories not seen during training. This technology excels in open-world scenarios and brings new possibilities to fields such as autonomous driving, intelligent robots, and dynamic environment perception.</td>
    <td>
    <ul>
        <li>Segmentation and path planning of unknown objects in autonomous driving</li>
        <li>Scene understanding by intelligent robots in unknown environments</li>
        <li>Real-time semantic segmentation and analysis in dynamic scenes</li>
    </ul>
    </td>
</tr>
<tr>
    <td>Rotated Object Detection</td>
    <td>Rotated Object Detection</td>
    <td>Not Available</td>
    <td>Rotated object detection is an important technology in the field of computer vision, focusing on detecting and locating objects with arbitrary orientations in images. Unlike traditional object detection methods (which usually assume objects are horizontal or vertical), rotated object detection can handle objects at any rotation angle, thus more accurately identifying and locating targets. By introducing oriented bounding boxes (OBB) and improved deep learning algorithms, this technology performs well in complex scenes such as aerial images, satellite images, and traffic sign detection in autonomous driving.</td>
    <td>
    <ul>
        <li>Target recognition and localization in aerial images</li>
        <li>Rotated traffic sign detection in autonomous driving</li>
        <li>Infrastructure detection in satellite images</li>
    </ul>
    </td>
</tr>
<tr>
    <td rowspan="2">Document Image Preprocessing</td>
    <td>Document Image Orientation Classification</td>
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
    <td>Text Image Rectification</td>
</tr>
<tr>
    <td>Multilingual Speech Recognition</td>
    <td>Multilingual Speech Recognition</td>
    <td>Not Available</td>
    <td>Multilingual speech recognition is an advanced speech processing technology that aims to automatically identify and transcribe speech signals in multiple languages to achieve efficient information extraction and communication. Compared to single-language speech recognition, multilingual speech recognition needs to handle differences in pronunciation, grammar, and vocabulary across languages, thus requiring more powerful models and richer language resources. Through deep learning and large-scale multilingual data training, this technology can recognize speech content in multiple languages in real-time and is widely used in intelligent translation, voice assistants, and multilingual customer service.</td>
    <td>
    <ul>
        <li>Multilingual interaction in intelligent voice assistants</li>
        <li>Real-time speech translation in international conferences</li>
        <li>Multilingual voice customer service systems</li>
    </ul>
    </td>
</tr>
<tr>
    <td>General Video Classification</td>
    <td>Video Classification</td>
    <td>Not Available</td>
    <td>Video classification is an important task in the field of computer vision, aiming to automatically analyze and identify the semantic categories of video content. Through deep learning models, video classification technology can extract spatiotemporal features from video frame sequences to accurately classify the themes, scenes, or activities in the video. This technology is widely used in video content recommendation, video surveillance analysis, intelligent media management, and video retrieval.</td>
    <td>
    <ul>
        <li>Content recommendation and classification in video platforms</li>
        <li>Abnormal behavior recognition in security surveillance</li>
        <li>Automatic classification and management of intelligent media libraries</li>
    </ul>
    </td>
</tr>
<tr>
    <td>General Video Detection</td>
    <td>Video Detection</td>
    <td>Not Available</td>
    <td>Video detection is a key technology in the field of computer vision, focusing on real-time or offline analysis of video content to identify and locate target objects and events in the video. By combining deep learning and object detection algorithms, video detection technology can handle complex dynamic scenes, detecting objects, people, behaviors, and abnormal events in the video. This technology has broad application prospects in intelligent security, traffic monitoring, sports analysis, and video content review.</td>
    <td>
    <ul>
        <li>Intrusion detection and alarm in intelligent security systems</li>
        <li>Vehicle detection and violation recognition in traffic monitoring</li>
        <li>Athlete behavior analysis in sports events</li>
    </ul>
    </td>
</tr>
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
