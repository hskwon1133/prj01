from ultralytics import YOLO

if __name__ == '__main__': #처음 실행할 때만 실행되었으면 좋겠다. (multi process 일때 무한 루프로 실행을 해줘라)
    model = YOLO('yolo26n.pt') #multi process 방식 /

    model.train(
        data='dataset/data.yaml',
        epochs=100,
        imgsz=(640, 640),
        batch=16, #16장씩
        name='jh_detector'
    )

