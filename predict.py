from ultralytics import YOLO

model = YOLO('./best.pt')
results = model('test_images', conf=0.831, save=True)

for r in results:
    print(r.to_json())

# 자바스크립트에서 요청하면 json 파일에서 읽혀야 함.

# 과제. 분류모델 만들기 사진 90장 똑같이 만들기. 내가 원하는 걸로