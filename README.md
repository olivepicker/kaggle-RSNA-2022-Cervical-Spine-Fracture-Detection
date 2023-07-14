## 🦴Kaggle RSNA 2022 Cervical Spine Fracture Detection🦴, 37th placed solution


__*My method is 2-Stage approaching.*__

- Stage 1 : Predict Bounding Box. </br>
- Stage 2 : Predict probabilities of fracture and position for each vertebrae.
- Finally, weighted averaging is used to obtain the probability of fracture at the position of each patient's C1-C7 vertebrae.</br></br>


### 1st Stage : Bounding Box Prediction </br>

`Data Preprocessing`
* CT Windowing [950, 1900]
  </br> </br>

`Hyperparameters`
* loss_function : L1Loss </br>
* optimizer : Adam </br>
* learning_rate : 1e-5</br>

</br>

### 2nd Stage : Image Level Prediction </br>

`Data Preprocessing`
</br>
* Set the three CT windowing values. I refered this paper https://arxiv.org/abs/2010.13336 </br>
  soft tissue : [300, 80] / gross bone : [650, 400] / standard bone : [1800, 500]</br>
* Some images are larger than 512, It resized to 512 </br>
* When I used Otsu's Threshold on gross bone channel, CV and LB scores are significantly increased.</br>
</br>

`Hyperparameters`

* loss_function : BCEWithLogitsLoss </br>
* optimizer : Adam with OneCycleLR scheduler </br>
* learning_rate : 1e-5</br>

</br>

`Highest Submissions`
* seresnext-101 * 1 + seresnext-50 * 1 
* Whole data trained, lb public 0.4086, lb private 0.4564 
</br>

`Not Worked`
* Changing CT windowing values 
* Implementing Feature Extraction -> LSTM stage, but failed.

</br>

`What I Learned`
* Stack adjacent slices which called Appian's windowing
* Approaches of Multi-Stage archtecture
* LSTM for Image data

</br>

`...`
* Many top solutions have proposed Segmentation model for vertebrae position -> Image Feature Extraction -> LSTM
* This 3-stage approaches are really works well, and I learned a lot from their code.
* I didn't have enough time and difficult to understand this data, which hindered me for making various attempts.
* But it was fun.
</br>
