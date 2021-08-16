# Action recognition using ConvLSTM
 Simple video-based action recognition pipeline using [ConvLSTM](https://paperswithcode.com/method/convlstm#:~:text=ConvLSTM%20is%20a%20type%20of,states%20of%20its%20local%20neighbors.)

## Dependencies
- Tensorflow 2.3
- OpenCV
- keras-video-generators

## Network
<p align="center">
<img src="https://github.com/farhantandia/ConvLSTM-action-recognition/blob/master/network.jpg"><br>
</p>
1. 3D input images
2. Multiple stacks of ConvLSTM along with batch normalization layer and with dropout 20%
3. Dense layers=> Final NN has 9.8M parameters with accuracy by 94.97%.

