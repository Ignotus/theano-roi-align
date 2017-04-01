# RoIAlign Operation for Theano

This is a fork of the [RoIPooling implementation on Theano](https://github.com/ddtm/theano-roi-pooling) for the RoIAlign operation
described in the paper "[Mask R-CNN](https://arxiv.org/pdf/1703.06870v1.pdf)" by He et al.

### Contributing

Currently only RoIAlign with max pooling aggregation has been implemented. However,
rewritting it to average pooling is straightforward. Please be free to send a pull
request if you have such.

### License

The source code is distributed under the same [BSD license](LICENSE) as the
original theano-roi-pooling code. Please refer to us if you find this code useful.
