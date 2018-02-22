from sklearn.datasets import fetch_mldata
import os
custom_data_home = '../data'
mnist = fetch_mldata('MNIST original', data_home=custom_data_home)
print(os.listdir(os.path.join(custom_data_home, 'mldata')))
