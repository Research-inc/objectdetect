from ultralytics import YOLO
from PIL import Image

def detectWithSearchName(input_image, search_name="book"):
    # Load a pretrained YOLOv8n-seg Segment model
    model = YOLO('yolov8n.pt')

    search_ind =  list(model.names.keys())[list(model.names.values()).index(search_name)]

    # Run inference on an image
    results = model(input_image, classes=search_ind)

    # We extract the detected object's x,y coordinates and bounded box width and height
    x0  = results[0].boxes.xywh.numpy()[0][0]
    y0  = results[0].boxes.xywh.numpy()[0][1]
    w   = results[0].boxes.xywh.numpy()[0][2]
    h = results[0].boxes.xywh.numpy()[0][3]

    # In the next step we find the mid point of detected object with respect to the overall image
    objx = x0 + (w / 2)
    objy = y0 + (h / 2)

    # We determine the width and height of the original image
    orig_height = results[0].boxes.orig_shape[0]
    orig_width = results[0].boxes.orig_shape[1]

    # We determine center of the original image. We use this center point as the origin for the cartesian coordinate system
    origin_x = orig_width/2
    origin_y = orig_height/2

    # We use the origin_x and origin_y along with objx, objy to determine of the location of object across four quadrants.
    quad = None

    if origin_x <= objx and origin_y >= objy:
        quad = "first"

    elif origin_x < objx and origin_y < objy:
        quad = "second"

    elif origin_x > objx and origin_y < objy:
        quad = "third"

    elif origin_x > objx and origin_y > objy:
        quad = "fourth"

    # Process results list
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save('result.jpg')  # save image

    return quad