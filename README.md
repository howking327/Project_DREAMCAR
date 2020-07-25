# Project_DREAMCAR

운전하는 상황에서 감정분석 및 졸음운전에 따른 음악추천으로 운전 집중도 향상과 다양한 음악 제공을 위한 딥러닝 모델 팀 프로젝트입니다.  
(2020/04/05 ~ 2020/06/05)

### ▶ Concept
운전자의 음악 감상 비중이 늘어나고 큐레이션 서비스가 확대되는 상황에 따라 운전자가 운전 중 직접 조작하는 단계를 줄이고 졸음 운전으로부터 
일시적인 예방을 통해 운전 집중도를 높이는 방안을 마련하기 위한 목적으로 해당 프로젝트를 진행하였습니다.

### ▶ Tool
- **Language** : Python  
- **IDE** : Pycharm / Colab / Jupyter Notebook
- **Detecting Tool** : OpenCV / Dlib
- **Web Tool** : HTML / CSS / JS
- **Framework** : Tensorflow / Keras / Scikit-learn / Flask(Web Framework) 

## Collaborators
- [howking327](https://github.com/howking327)
- [yiy829](https://github.com/yiy829)
- [jaden7856](https://github.com/jaden7856)


## Data Set
- 감정 분석 데이터 셋
![1](https://user-images.githubusercontent.com/57980363/88459270-fe1b2c80-cece-11ea-9cc3-55c2ed76ffbe.png)

- 음악 플레이리스트 추천 데이터 셋
![2](https://user-images.githubusercontent.com/57980363/88459273-01aeb380-cecf-11ea-9d82-59be6bacedba.png)

### ▶ Project
![3](https://user-images.githubusercontent.com/57980363/88459274-04110d80-cecf-11ea-9186-a0229220a9f8.png)

본 프로젝트는 크게 3가지 서비스를 통합한 프로젝트로서 1단계 졸음판단, 2단계 감정분석, 3단계 추천시스템 순서로 각각의 완료된 분석모델을 통합하는 방향으로 개발을 진행했습니다. 
각 시스템의 처리시간을 최소화하기 위해 모델을 단순화시켰습니다. 카메라를 통한 실시간 분석이라는 측면에서 모델의 깊이가 깊고 무거워지면 그만큼 연산속도와 메모리 사용량이 증가해 
실시간 분석의 속도 측면에서 실효성이 떨어지기 때문입니다. 사용한 프레임워크는 대표적으로 Flask, Keras, Tensorflow, Scikit-learn을 사용하였습니다.

## Drowsy Detect
![4](https://user-images.githubusercontent.com/57980363/88459275-06736780-cecf-11ea-83de-f837a21ffb33.png)

1단계 졸음 판단입니다. 카메라를 통해 영상 데이터가 들어오게 되면 안면에 좌표를 찍습니다. 
눈과 입 좌표값을 사용하여 결과에 따라 졸음/정상상태를 판단하고 졸음판단일 경우 경고음을 울리게 됩니다.
졸음상태는 눈과 입의 움직임 변화량을 판단하게 되는데, 먼저 눈 종횡비를 측정합니다. 눈이 감기거나 자주 깜빡이게 되면 졸음 상태라 간주합니다. 
또한 눈 종횡비를 보조하기 위해 눈이 감기게 되면 동공원형도가 유지되지 않는 점을 수치로 사용했습니다. 
입의 경우에는 눈 종횡비와 반대로 하품으로 인해 입이 벌어질 때 값이 커지는 것을 이용하여 입의 변화로 졸음을 판단합니다. 
또한 위 값들을 보조하기 위해서 눈 종횡비와 입 종횡비의 비율 수치를 추가로 적용합니다.
그리고 저희는 고정된 운전자가 아닌 불특정 다수를 대상으로 판단해야 하기 때문에 사람마다 눈의 크기가 다른 것을 고려하여 
처음 5프레임동안 측정된 눈 종횡비 값의 평균을 기준 값으로 그에 따른 변화량을 통해서 사람마다 유동적인 대응을 할 수 있도록 했습니다.

## Emotion Classification
![5](https://user-images.githubusercontent.com/57980363/88459278-083d2b00-cecf-11ea-91fc-03eb5311b8a1.png)
![6](https://user-images.githubusercontent.com/57980363/88459279-0a06ee80-cecf-11ea-8901-28be1819c415.png)

다음은 2단계 감정 분석입니다. 졸음 판단과 마찬가지로 영상 데이터가 들어오면 오브젝트를 인식하는 Haarcascades라이브러리를 이용해 안면을 인식하고 
인식된 프레임을 프레임 단위로 이미지 전처리를 통해 단순화 시킨 딥러닝 모델에 입력합니다. 모델은 BatchNorm을 사용한 모델과 VGG16을 절반으로 축소한
2가지 모델 중 판단 Accuracy가 높고 실시간 감정 분석 속도가 빠른 VGG 단순화 모델을 채택하여 적용했습니다. 
모델은 들어온 이미지를 분석하여 운전자에게 나타날 수 있는 대표적인 감정인 행복, 화남, 일반 3가지 감정으로 분류합니다.

## Recommendation Algorithm
![7](https://user-images.githubusercontent.com/57980363/88459281-0bd0b200-cecf-11ea-81c7-a5de07ac561f.png)
![8](https://user-images.githubusercontent.com/57980363/88459283-0d9a7580-cecf-11ea-8359-70a745fe6108.png)

마지막 3단계 추천 알고리즘입니다. 1단계와 2단계의 결과에 따라서 1)졸음의 경우 졸음을 최종상태로 2)깨어있는 상태에는 3가지 감정 중 판단된 결과가 최종 상태로 전달됩니다. 
전달된 운전자상태는 웹크롤링과 시스템 정보를 통해 가져온 여러 실시간 정보를 종합하여 추천 모델을 통해 음악리스트가 출력되고 해당 리스트의 음악을 재생하게 됩니다.
추천 알고리즘의 경우 데이터셋의 특성과 DB를 통한 사용자 정보 구축이 되어 있지 않아 컨텐츠 기반 필터링을 통해 콜드 스타트 문제를 피할 수 있도록 했습니다.

## Web Programing
![9](https://user-images.githubusercontent.com/57980363/88459284-0f643900-cecf-11ea-972a-45fad93efb5f.png)

위 3가지 단계를 통한 과정을 통합하고 Flask 웹 프레임워크를 이용하여 웹을 통해 분석 모델을 실행하고 음악을 추천 받아 재생할 수 있도록 했습니다.
프로그램의 전반적인 과정과 구조는 위와 같습니다.

## Source Code
- argument_parser.py : 졸음 판단 기준 값이나 외부 라이브러리 경로 등 변경 가능한 변수들을 통합
- cnn_model.py : 감정분석 모델과 커스텀 함수 생성
- detecting.py : 카메라를 통한 실시간 분석 모델
- drowsy_landmark.py : 졸음 측정 시 기준이 되는 landmark 좌표 값을 설정
- recommend.py : 추천 알고리즘 컨텐츠를 기반으로 하여 코사인 유사도를 통해 플레이리스트 도출
- result_toweb.py : 도출된 플레이리스트를 음원 사이트를 통해 검색하고 재생할 수 있도록 설정
- web_main.py : Flask를 통한 웹 프로그램 실행

## Improvements
ⓐ 안면인식을 활용하여 운전자 상태/행동 데이터 정보를 DB에 저장 합니다. 개인별 안면 인식을 통해서 운전자를 식별 및 상태/행동 데이터를 저장하며, 
차량 내에서 일어나는 운전자의 다양한 상태와 행동을 분석하여 안전운전에 도움을 줄 것으로 기대합니다.

ⓑ DB에 정보가 쌓이면 “콜드 스타트” 문제로 인해 사용하지 못한 collaborative filtering 추천모델을 적용하여 보다 향상된 추천 리스트를 제공할 수 있습니다. 
또한 음악 서비스 기업과 협업을 통해서 기존의 웹페이지를 통한 음악 재생이 아닌 DB에 저장된 운전자의 로그인 정보를 통해서 자체적인 음악재생 플레이어로 프로그램을 구성해야 할 것입니다.

ⓒ 자동차에 내장되어 있는 분석 모델도 더 좋은 모델을 개발하고 학습시켜서, 개선된 모델과 학습된 파라미터를 자동차로 전송하는 방식으로 지속적인 업데이트를 해야 할 것입니다.

