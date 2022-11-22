import numpy as np
np.set_printoptions(precision=2, linewidth=150, floatmode='fixed')

def print_metric(metric, metric_name='ACC'):
    print('#######################################')
    print("Metric={}".format(metric_name))
    num_types = metric.shape[0]
    mean_metric_str = ''
    for type_i in range(num_types):
        metric_i = metric[type_i, :]
        mean_metric_str += "Type{}: {:.2f}\t".format(type_i+1, np.mean(metric_i))
    print(mean_metric_str)
    print('==={} of Each Angle==='.format(metric_name))
    for type_i in range(num_types):
        metric_i = metric[type_i, :]
        print("Type{}: {}".format(type_i+1, metric_i))     
    print('#######################################')