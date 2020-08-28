from utils import *
from image import letter_image
from Darknet import Darknet


def detect(cfgfile, weightfile, imgfile, namefile):
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    use_cuda = False  # if torch.cuda.is_available() else False
    if use_cuda:
        m.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = letter_image(img, m.width, m.height)

    start = time.time()
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda=use_cuda)

    finish = time.time()
    print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = read_class_names(namefile)
    plot_boxes(img, boxes, 'predictions.jpg', class_names)


if __name__ == '__main__':
    cfgfile = 'data/yolo_v3.cfg'
    weightfile = '../yolov3_pth/yolo_v3_250epochs_81.4.pth'
    #weightfile = './yolo_v3_250epochs_81.4.pth'
    imgfile = "./imgs/dog.jpg"
    namefile = 'data/voc.names'
    detect(cfgfile, weightfile, imgfile, namefile)
