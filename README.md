# Occluded-Facial-Expression-Recognition
Recognizing facial expressions in occluded images using non-occluded images as privileged information.  
Related research paper: [Occluded Facial Expression Recognition Enhanced through Privileged Information](https://dl.acm.org/doi/10.1145/3343031.3351049#:~:text=In%20this%20paper%2C%20we%20propose,but%20not%20required%20during%20testing.)

## Datasets
Three benchmark datasets are used.

- [RAF-DB](http://www.whdeng.cn/raf/model1.html) 
- [AffectNet](http://mohammadmahoor.com/affectnet/)
- [FED-RO](https://www.semanticscholar.org/paper/Occlusion-Aware-Facial-Expression-Recognition-Using-Li-Zeng/b5bb7e12a15b57b4d307e742da127a74596d0c7c)


For RAF-DB and AffectNet, synthetic occlusions are imposed to generate occluded images.

## Model Architecture
- Two deep convolutional neural networks with the same architecture are built in the framework. 
- The first one performs expression recognition from occluded images. 
- The second one performs expression recognition from non-occluded images and is used as a guide. 
- These two networks are first pre-trained with the supervised multi-class cross-entropy losses. 
- After pretraining, parameters of the non-occluded network are fixed and the occluded net is further fine-tuned under the guidance of the non-occluded net. 
- Non-occluded net provides guidance to the occluded net in the feature space and the label space
- Guidance in label space is provided using a similarity constraint and a loss inequality regularizer.
- Guidance in feature space is provided using an adversarial loss and a reconstruction loss.


![model](https://user-images.githubusercontent.com/31109495/118502898-c0caa880-b747-11eb-98f0-8407f0313393.png)

## Results
Upon finetuning the occluded-net using privileged information, I observe a gain of 4.27% on RAF-DB, 3.06% on AffectNet, and 4.38% on the FED-RO dataset.

<table>
  <tr>
    <th rowspan="2"><b>Dataset</b></th>
    <th colspan="2"><b>Accuracy</b></th>
  </tr>
  <tr>
    <td><b>Base Model</b></td>
    <td><b>Privileged Information</b></td>
  </tr>
  <tr>
    <td>RAF-DB</td>
    <td>75.91%</td>
    <td>80.18%</td>
  </tr>
  <tr>
    <td>AffectNet</td>
    <td>54.06%</td>
    <td>57.12%</td>
  </tr>
  <tr>
    <td style="text-align:center">FED-RO</td>
    <td>64.94%</td>
    <td>69.32%</td>
  </tr>
</table>

