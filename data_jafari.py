import numpy as np

# Provided x and y values
x = [5.5058823529411764, 5.88235294117647, 6.188235294117647, 6.4941176470588236, 6.847058823529411, 7.152941176470588, 7.435294117647059, 7.670588235294118, 7.929411764705883, 8.16470588235294, 8.376470588235295, 5.764705882352941, 6.188235294117647, 6.588235294117647, 6.964705882352941, 7.317647058823529, 7.8352941176470585, 8.329411764705883, 8.68235294117647, 9.152941176470588, 9.670588235294117, 10.211764705882352, 10.729411764705882, 11.223529411764705, 11.694117647058823, 12.235294117647058, 12.729411764705882]
y = [-0.0425531914893617, -0.3829787234042553, -0.851063829787234, -1.2765957446808511, -1.7446808510638296, -2.2127659574468086, -2.723404255319149, -3.4468085106382977, -4.170212765957447, -4.8510638297872335, -5.48936170212766, 0, 0.3829787234042553, 0.851063829787234, 1.3191489361702127, 1.7872340425531914, 2.5106382978723403, 3.276595744680851, 4, 4.8510638297872335, 5.531914893617021, 6.51063829787234, 7.531914893617021, 8.51063829787234, 9.446808510638297, 10.297872340425531, 11.361702127659575]
# Combine into a single NumPy array
jafari_913Pa_40heat = np.column_stack((x, y))