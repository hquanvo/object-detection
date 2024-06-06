import numpy as np

from coco_object_detection import Darknet, load_classes, prep_image, write_results
import torch
from torch.autograd import Variable
import cv2
from sort import Sort



def camera_evaluate():

    tracker = Sort()

    video = cv2.VideoCapture(0)

    model = Darknet("cfg/yolov3.cfg")
    model.load_weights("yolov3.weights")

    classes = load_classes("./class_names.txt")

    num_classes = len(classes)

    confidence = 0.5
    nms_thesh = 0.4

    model.net_info["height"] = 416
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32


    CUDA = torch.cuda.is_available()

    if CUDA:
        model.cuda()

    model.eval()

    while True:
        retval, frame = video.read()


        if retval:
            img = prep_image(frame, inp_dim)
            im_dim = frame.shape[1], frame.shape[0]
            im_dim = torch.FloatTensor(im_dim).repeat(1,2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = model(img, CUDA)

            output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
            output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

            output[:, 1:5] /= scaling_factor


            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

            detections = []
            for obj in output:
                detections.append([obj[1], obj[2], obj[3], obj[4], obj[5]])

            detections = np.array(detections)
            tracked_objs = tracker.update(detections)
            def write(x, y, results):
                c1 = tuple((int(y[0].item()), int(y[1].item())))
                c2 = tuple((int(y[2].item()), int(y[3].item())))
                print(y)

                img = results[int(x[0])]
                cls = int(x[-1])
                color = ((cls + 30) * (cls % 3), (cls + 30) * ((cls + 1) % 3), (cls + 30) * ((cls + 2) % 3))
                id = int(y[-1])
                label = "{0},ID:{1}".format(classes[cls], id)
                cv2.rectangle(img, c1, c2, color, 1)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
                cv2.rectangle(img, c1, c2, color, -1)
                cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
                return img

            list(map(lambda x, y: write(x,y, [frame]), output, tracked_objs))



            cv2.imshow("Video", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else: break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_evaluate()