import sys
import os

def _scan_image_tree(dir_path, img_list):
    for i, name in enumerate(os.listdir(dir_path)):
        full_path = os.path.join(dir_path, name)
        if os.path.isdir(full_path):
            _scan_image_tree(full_path, img_list)
        else:
            img = full_path
            if(img.lower().endswith('.jpg') or img.lower().endswith('.png')
                    or img.lower().endswith('.gif')
                    or img.lower().endswith('.jpeg')
                    or img.lower().endswith('.jpeg2000')
                    or img.lower().endswith('.tif')
                    or img.lower().endswith('.psg')
                    or img.lower().endswith('.swf')
                    or img.lower().endswith('.svg')
                    or img.lower().endswith('.bmp')):
                img_list += [img]
            if len(img_list)%100 == 0:
                sys.stdout.flush()
                sys.stdout.write('\r #img of scan: %d'%(len(img_list)))


def scan_image_tree(dir_path):
    img_list = []
    _scan_image_tree(dir_path, img_list)
    sys.stdout.write('\n')
    return img_list


def load_model(model):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Error load ' + model)
        exit(0)


