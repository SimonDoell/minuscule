# Minuscule

_This is a work in progress_

## ToDo:
- learning rate schedulers (e.g. cosine weight decay)
- cached matricis inside the layers to avoid re-allocation inside forward/backward
- dropout layer
- add the options for a differnet optimizer using tagged enums (e.g. SGD, RMSProp)
- gradient clipping
- residual connections?
- different activation functions
- arena allocator
- model serialization (saving and loading to/from file)