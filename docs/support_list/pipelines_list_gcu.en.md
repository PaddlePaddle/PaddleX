---
comments: true
---

# PaddleX Pipelines (GCU)

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
</table>

## 2. Featured Pipelines
Not supported yet, please stay tuned!
