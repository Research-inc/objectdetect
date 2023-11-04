from ultralytics import YOLO
from PIL import Image

def detectWithSearchName(input_image, search_name="book"):
    # Load a pretrained YOLOv8n-seg Segment model
    model = YOLO('yolov8n.pt')

    search_ind =  list(model.names.keys())[list(model.names.values()).index(search_name)]

    # Run inference on an image
    results = model(input_image, classes=search_ind)

    # Process results list
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save('result.jpg')  # save image

#detectWithSearchName('object.jpg', "car")