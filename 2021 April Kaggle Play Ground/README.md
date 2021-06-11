# 2021 April Kaggle Play Ground

![titanic1](https://user-images.githubusercontent.com/76901290/121700683-1d34a400-cb0b-11eb-9d15-43434e1943e3.png)

- 일정: 2021.04.01 - 2021.04.30
- 주최 및 주관: Kaggle
- 데이터 출처: https://www.kaggle.com/c/titanic/data
- The dataset is used for this competition is synthetic but based on a real dataset (in this case, the actual Titanic data!) and generated using a CTGAN.
- 참고: https://www.kaggle.com/c/tabular-playground-series-apr-2021/

## 개요

1. 과제

- Your task is to predict whether or not a passenger survived the sinking of the Synthanic (a synthetic, much larger dataset based on the actual Titanic dataset).

2. 상세 설명

- For each PasengerId row in the test set, you must predict a 0 or 1 value for the Survived target.

3. 평가 기준

- You should submit a csv file with exactly 100,000 rows plus a header row. Your submission will show an error if you have extra columns or extra rows.

- The file should have exactly 2 columns:
PassengerId (sorted in any order)
Survived (contains your binary predictions: 1 for survived, 0 for deceased)

4. 팀 구성

- 작업툴: Kaggle Notebook

- 인원: 4명

- 주요 업무: EDA, 모델링, PPT 제작

- 기간: 2021.04.19 - 2021.04.30

- 데이터

- The dataset is used for this competition is synthetic but based on a real dataset (in this case, the actual Titanic data!) and generated using a CTGAN.

- The data has been split into two groups:
  training set (train.csv)
  test set (test.csv)

- The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger.

- The test set should be used to see how well your model performs on unseen data.
   
![titanic2](https://user-images.githubusercontent.com/76901290/121700748-2de51a00-cb0b-11eb-97e3-a82029ffd8b3.png)