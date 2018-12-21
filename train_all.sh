# python car_classifier.py --model_name densenet121 --optimizer SGD --re_train True --lr 0.001
# python car_classifier.py --model_name densenet121 --optimizer Adam --re_train True --lr 0.001
# python car_classifier.py --model_name resnet18 --optimizer SGD --re_train True --lr 0.001
# python car_classifier.py --model_name resnet18 --optimizer Adam --re_train True --lr 0.001
# python car_classifier.py --model_name resnet34 --optimizer SGD --re_train True --lr 0.001
# python car_classifier.py --model_name resnet34 --optimizer Adam --re_train True --lr 0.001
# python car_classifier.py --model_name inception_v3 --optimizer SGD --re_train True --lr 0.001
# python car_classifier.py --model_name inception_v3 --optimizer Adam --re_train True --lr 0.001

python car_classifier.py --model_name resnet18 --optimizer SGD --lr 0.0001
python car_classifier.py --model_name resnet34 --optimizer SGD --lr 0.0001