---
comments: true
---

# PaddleX产线列表(GCU)

## 1、基础产线

<table>
    <tr>
        <th width="10%">产线名称</th>
        <th width="10%">产线模块</th>
        <th width="10%">星河社区体验地址</th>
        <th width="50%">产线介绍</th>
        <th width="20%">适用场景</th>
    </tr>
  <tr>
    <td>通用图像分类</td>
    <td>图像分类</td>
    <td><a href="https://aistudio.baidu.com/community/app/100061/webUI">在线体验</a></td>
    <td>图像分类是一种将图像分配到预定义类别的技术。它广泛应用于物体识别、场景理解和自动标注等领域。图像分类可以识别各种物体，如动物、植物、交通标志等，并根据其特征将其归类。通过使用深度学习模型，图像分类能够自动提取图像特征并进行准确分类。</td>
    <td>
    <ul>
        <li>商品图片的自动分类和识别</li>
        <li>流水线上不合格产品的实时监控</li>
        <li>安防监控中人员的识别</li>
      </ul>
  </tr>
  <tr>
    <td>通用目标检测</td>
    <td>目标检测</td>
    <td><a href="https://aistudio.baidu.com/community/app/70230/webUI">在线体验</a></td>
    <td>目标检测旨在识别图像或视频中多个对象的类别及其位置，通过生成边界框来标记这些对象。与简单的图像分类不同，目标检测不仅需要识别出图像中有哪些物体，例如人、车和动物等，还需要准确地确定每个物体在图像中的具体位置，通常以矩形框的形式表示。该技术广泛应用于自动驾驶、监控系统和智能相册等领域，依赖于深度学习模型（如YOLO、Faster R-CNN等），这些模型能够高效地提取特征并进行实时检测，显著提升了计算机对图像内容理解的能力。</td>
    <td>
      <ul>
        <li>视频监控中移动物体的跟踪</li>
        <li>自动驾驶中车辆的检测</li>
        <li>工业制造中缺陷产品的检测</li>
        <li>零售业中货架商品的检测</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td rowspan = 2>通用OCR</td>
    <td>文本检测</td>
    <td rowspan = 2><a href="https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent">在线体验</a></td>
    <td rowspan = 2>OCR（光学字符识别，Optical Character Recognition）是一种将图像中的文字转换为可编辑文本的技术。它广泛应用于文档数字化、信息提取和数据处理等领域。OCR 可以识别印刷文本、手写文本，甚至某些类型的字体和符号。 通用 OCR 产线用于解决文字识别任务，提取图片中的文字信息以文本形式输出，PP-OCRv4 是一个端到端 OCR 串联系统，可实现 CPU 上毫秒级的文本内容精准预测，在通用场景上达到开源SOTA。基于该项目，产学研界多方开发者已快速落地多个 OCR 应用，使用场景覆盖通用、制造、金融、交通等各个领域。</td>
    <td rowspan = 2>
    <ul>
        <li>智能安防中车牌号</li>
        <li>门牌号等信息的识别</li>
        <li>纸质文档的数字化</li>
        <li>文化遗产中古代文字的识别</li>
      </ul>
      </td>
  </tr>
  <tr>
    <td>文本识别</td>
  </tr>
</table>

## 2、特色产线
暂不支持，敬请期待！
